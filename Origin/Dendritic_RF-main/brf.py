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

class ZIF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, gama):
        out = (input > 0).float()
        L = torch.tensor([gama])
        ctx.save_for_backward(input, out, L)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (input, out, others) = ctx.saved_tensors
        gama = others[0].item()
        grad_input = grad_output.clone()
        tmp = (1 / gama) * (1 / gama) * ((gama - input.abs()).clamp(min=0))
        grad_input = grad_input * tmp
        return grad_input, None


class LIFSpike(nn.Module):
    def __init__(self, thresh=1, tau=0., gama=1.0):
        super(LIFSpike, self).__init__()
        self.act = ZIF.apply
        self.thresh = thresh
        self.tau = tau
        self.gama = gama

    def forward(self, x):
        mem = 0
        spike_pot = []
        mem_pot = []
        T = x.shape[2]
        for t in range(T):
            mem = mem * self.tau + x[..., t]
            spike = self.act(mem - self.thresh, self.gama)
            mem = (1 - spike) * mem
            mem_pot.append(mem)
            spike_pot.append(spike)
        return torch.stack(spike_pot, dim=2), torch.stack(mem_pot, dim=2)

# ========== Step 1: 生成稀疏脉冲序列 ==========
def generate_sparse_pulse_sequence(length=100, num_pulses=40, amplitude_range=(0.1, 1.0)):
    sequence = np.zeros(length)
    indices = np.random.choice(length, num_pulses, replace=False)
    amplitudes = np.random.uniform(amplitude_range[0], amplitude_range[1], num_pulses)
    sequence[indices] = amplitudes
    return sequence

# ========== Step 2: 完整绘图函数（静态+动态合并） ==========
def plot_with_dynamic_threshold(pulse_seq, potential_seq, spike_seq, threshold_seq, dynamic_spike_seq, save_path=None):
    T = len(pulse_seq)
    fig, axs = plt.subplots(3, 1, figsize=(10, 4.5), sharex=True, height_ratios=[1, 2, 1])

    # 输入脉冲
    for i, amp in enumerate(pulse_seq):
        if amp > 0:
            axs[0].vlines(i, 0, amp, color='royalblue', linewidth=1.0)
    axs[0].set_ylim(0, 1.05)
    axs[0].set_yticks([0, 1])
    axs[0].set_ylabel("Input\nRandom", fontsize=9, rotation=0, labelpad=25, weight='bold')
    axs[0].grid(axis='y', linestyle='--', linewidth=1)
    axs[0].tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    axs[0].set_title("LIF Sequential Computation", fontsize=12, weight='bold')

    # 膜电位 + 动态阈值
    axs[1].plot(potential_seq, label="Membrane Potential", color='steelblue')
    axs[1].plot(threshold_seq, label="Dynamic Threshold", color='red')
    # axs[1].set_ylim(0, max(2.0, np.max(potential_seq) + 0.5))
    axs[1].set_ylabel("Membrane\nPotential", fontsize=9, rotation=0, labelpad=30, weight='bold')
    # axs[1].legend(loc='upper right', fontsize=8)
    axs[1].grid(axis='y', linestyle='--', linewidth=1)
    axs[1].tick_params(axis='x', which='both', bottom=False, labelbottom=False)

    # # 合并绘制静态 + 动态发放
    for i, amp in enumerate(spike_seq):
        if amp > 0:
            axs[2].vlines(i, 0, amp, color='darkorange', linewidth=1.0, label='Static Threshold' if i == 0 else None)
    for i, amp in enumerate(dynamic_spike_seq):
        if amp > 0:
            axs[2].vlines(i, 0, amp, color='royalblue', linewidth=1.0, label='Dynamic Threshold' if i == 0 else None)

    axs[2].set_ylim(0, 1.05)
    axs[2].set_yticks([0, 1])
    axs[2].set_ylabel("Output\nSpiking", fontsize=9, rotation=0, labelpad=25, weight='bold')
    axs[2].set_xlabel("Timestep", fontsize=10)
    axs[2].legend(loc='upper right', fontsize=8)
    axs[2].grid(axis='y', linestyle='--', linewidth=1)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"Saved dynamic threshold plot to {save_path}")
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

    
    potential_sum = torch.cumsum(potential, dim=-1)
    shift_potential = torch.zeros_like(potential_sum)

    shift_potential[..., 1:] = potential_sum[..., :-1]
    threshold_dynamic_seq = potential_sum - torch.floor(shift_potential)
    # 静态阈值发放
    spike_seq = (potential_seq > 1).astype(float)
    spike_new = (threshold_dynamic_seq > 1).float()

    # 绘图
    plot_with_dynamic_threshold(
        pulse_seq, potential_seq, spike_seq,
        threshold_dynamic_seq[0, 1].detach().cpu().numpy(), spike_new[0, 1].detach().cpu().numpy(),
        save_path="output/lif_dynamic_threshold_plot.png"
    )