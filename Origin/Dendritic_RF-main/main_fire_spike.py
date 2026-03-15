import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.newVGGSNN import S4Model

# 设置随机种子确保可复现
np.random.seed(200)
torch.manual_seed(200)

# ========== Step 1: 生成稀疏脉冲序列 ==========
def generate_sparse_pulse_sequence(length=100, num_pulses=40, amplitude_range=(0.1, 1.0)):
    sequence = np.zeros(length)
    indices = np.random.choice(length, num_pulses, replace=False)
    amplitudes = np.random.uniform(amplitude_range[0], amplitude_range[1], num_pulses)
    sequence[indices] = amplitudes
    return sequence

# Re-import libraries and re-define function after state reset
import numpy as np
import matplotlib.pyplot as plt
import os

def plot_static_vs_dynamic_split(pulse_seq, potential_seq, spike_static, threshold_seq_dynamic, spike_dynamic, save_path=None):
    """
    左边绘制静态阈值结果，右边绘制动态阈值结果
    """
    T = len(pulse_seq)
    # fig, axs = plt.subplots(3, 2, figsize=(20, 3), sharex='col', sharey='row', gridspec_kw={'wspace': 0.25, 'hspace': 0.4, 'height_ratios': [1, 1.5,1]})
    fig, axs = plt.subplots(
      3, 2,
      figsize=(20, 3),
      sharex='col',
      sharey='row',
      gridspec_kw={
          'height_ratios': [1, 1.5, 1]
      },
      constrained_layout=True
  )

    # fig, axs = plt.subplots(
    #     3, 2,
    #     figsize=(16, 3),
    #     sharex='col',
    #     sharey='row',
    #     gridspec_kw={'wspace': 0.3, 'hspace': 0.5, 'width_ratios': [1.2, 1.2]},
    #     constrained_layout=True
    # )
    # --- Row 0: Input ---
    for i, amp in enumerate(pulse_seq):
        if amp > 0:
            axs[0, 0].vlines(i, 0, amp, color='#2F5597', linewidth=3.0)
            axs[0, 1].vlines(i, 0, amp, color='#2F5597', linewidth=3.0)
    for ax in axs[0]:
        ax.set_ylim(0, 1.05)
        ax.set_yticks([0, 1])
        ax.set_ylabel("Input", fontsize=9)
        ax.grid(axis='y', linestyle='--', linewidth=0.5)
    axs[0, 0].set_title("Static Threshold")
    axs[0, 1].set_title("Dynamic Threshold")

    # --- Row 1: Membrane Potential ---
    axs[1, 0].plot(potential_seq, color='steelblue', linewidth=2.5)
    axs[1, 0].axhline(1.0, color='red', linestyle='--', label='Static Thresh', linewidth=2.5)
    axs[1, 1].plot(potential_seq, color='steelblue', linewidth=2.5)
    axs[1, 1].plot(threshold_seq_dynamic, color='red', linestyle='--', label='Dynamic Thresh', linewidth=2.5)
    for ax in axs[1]:
        ax.set_ylim(0, max(2.0, np.max(potential_seq) + 0.2))
        ax.set_ylabel("Mem.\nPotential", fontsize=9)
        ax.grid(axis='y', linestyle='--', linewidth=0.5)

    # --- Row 2: Spikes ---
    for i, amp in enumerate(spike_static):
        if amp > 0:
            axs[2, 0].vlines(i, 0, amp, color='darkorange', linewidth=3.0)
    for i, amp in enumerate(spike_dynamic):
        if amp > 0:
            axs[2, 1].vlines(i, 0, amp, color='#0D77C3', linewidth=3.0)
    for ax in axs[2]:
        ax.set_ylim(0, 1.05)
        ax.set_yticks([0, 1])
        ax.set_ylabel("Spikes", fontsize=9)
        ax.set_xlabel("Timestep")
        ax.grid(axis='y', linestyle='--', linewidth=0.5)

    # plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"Saved dual-panel plot to {save_path}")
        plt.close()
    else:
        plt.show()

class CausalConvSoftmaxWeight(nn.Module):
    def __init__(self, in_channels, kernel_size):
        super().__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size

        # 权重 shape: [kernel_size]，行和为1
        self.weight = torch.tensor([0.2500, 0.4192, 0.2194, 0.1113], dtype=torch.float32)
        self.padding = (kernel_size, 0)

    def forward(self, x):
        # 正确 repeat 到 depthwise 卷积所需的形状
        weight = self.weight.reshape(1, 1, -1).repeat(self.in_channels, 1, 1)  # [C,1,K]

        x_padded = F.pad(x, self.padding)
        out = F.conv1d(x_padded, weight, stride=1, groups=self.in_channels)
        return out[..., : -1]

    
# ========== Step 3: 模拟主流程 ==========
if __name__ == "__main__":
    seq_len = 100
    pulse_num = 40
    amplitude_rng = (0.1, 1)   

    # 输入信号
    pulse_seq = generate_sparse_pulse_sequence(seq_len, pulse_num, amplitude_rng)

    # 模型加载
    state_dict = torch.load('resnet-256-new.pth', map_location=torch.device('cpu'))
    model = S4Model(d_input=1, d_model=128)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    print(model)
    # 输入预处理
    pulse_tensor = torch.from_numpy(pulse_seq.astype(np.float32)).unsqueeze(0).unsqueeze(0)  # [1,1,T]
    pulse_tensor = pulse_tensor.repeat(1, 128, 1)  # [1,C,T]

    # 获取电位
    _, potential = model.layer1.neuron1(pulse_tensor)
    potential_seq = potential[0, 1].detach().numpy()

    # 静态阈值发放
    spike_seq = (potential_seq > 1).astype(float)

    # 动态阈值计算 + 动态发放
    # raw_weight = model.layer1.neuron1.conv1.raw_weight
    # weight = F.softmax(raw_weight, dim=-1)  # [C, K]
    # print(weight)
    conv1 = CausalConvSoftmaxWeight(kernel_size=4, in_channels=128)
    threshold_dynamic =conv1(model.layer1.neuron1.act1(potential - 1.)) + 1 
    threshold_dynamic_seq = threshold_dynamic[0, 1].detach().numpy()
    spike_dynamic = (potential_seq > threshold_dynamic_seq).astype(float)

    # 绘图
    plot_static_vs_dynamic_split(
        pulse_seq, potential_seq, spike_seq,
        threshold_dynamic_seq, spike_dynamic,
        save_path="output/lif_dynamic_threshold_plot.png"
    )