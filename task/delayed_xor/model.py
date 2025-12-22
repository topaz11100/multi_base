from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from neurons.CP_LIF_neuron import CP_LIF
from neurons.DH_SNN_neuron import readout_integrator_test, spike_dense_test_denri_wotanh_R
from neurons.TC_LIF_neuron import TCLIFNode
from neurons.TS_LIF_neuron import TSLIFNode
from util import count_parameters


class Model_DH(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, device: torch.device):
        super().__init__()
        self.device = device
        self.dense_1 = spike_dense_test_denri_wotanh_R(
            input_dim,
            hidden_dim,
            tau_minitializer="uniform",
            low_m=0,
            high_m=4,
            branch=1,
            vth=1,
            dt=1,
            device=device,
            bias=True,
        )
        self.dense_2 = readout_integrator_test(hidden_dim, output_dim, dt=1, device=device, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, time_steps, _ = x.shape
        self.dense_1.set_neuron_state(batch_size)
        self.dense_2.set_neuron_state(batch_size)

        outputs = []
        for t in range(time_steps):
            mem1, spike1 = self.dense_1(x[:, t, :])
            mem2 = self.dense_2(spike1)
            outputs.append(mem2.unsqueeze(1))
        return torch.cat(outputs, dim=1)

    def print_model_parameters(self) -> None:
        print(f"Model_DH parameters: {count_parameters(self)}")


class _FeedForwardLIF(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, device: torch.device):
        super().__init__()
        self.device = device
        self.fc_in = nn.Linear(input_dim, hidden_dim)
        self.readout = nn.Linear(hidden_dim, output_dim)

    def reset_state(self) -> None:  # pragma: no cover - to be implemented by subclass
        raise NotImplementedError

    def neuron_forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input comes as (B, T, C); convert to (T, B, C) for convenience
        b, t, _ = x.shape
        x_seq = self.fc_in(x.view(b * t, -1)).view(b, t, -1).permute(1, 0, 2)
        self.reset_state()
        spikes = self.neuron_forward(x_seq)
        out = self.readout(spikes)
        return out.permute(1, 0, 2)

    def print_model_parameters(self) -> None:
        print(f"{self.__class__.__name__} parameters: {count_parameters(self)}")


class Model_CP(_FeedForwardLIF):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, device: torch.device):
        super().__init__(input_dim, hidden_dim, output_dim, device)
        self.lif = CP_LIF((hidden_dim,))

    def reset_state(self) -> None:
        self.lif.reset_state()

    def neuron_forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lif(x)


class Model_TC(_FeedForwardLIF):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, device: torch.device):
        super().__init__(input_dim, hidden_dim, output_dim, device)
        self.node = TCLIFNode(step_mode="s")

    def reset_state(self) -> None:
        self.node.reset()

    def neuron_forward(self, x: torch.Tensor) -> torch.Tensor:
        spikes = []
        for step in range(x.shape[0]):
            spikes.append(self.node(x[step]))
        return torch.stack(spikes, dim=0)


class Model_TS(_FeedForwardLIF):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, device: torch.device):
        super().__init__(input_dim, hidden_dim, output_dim, device)
        self.node = TSLIFNode(step_mode="s")

    def reset_state(self) -> None:
        self.node.reset()

    def neuron_forward(self, x: torch.Tensor) -> torch.Tensor:
        spikes = []
        for step in range(x.shape[0]):
            spikes.append(self.node(x[step]))
        return torch.stack(spikes, dim=0)


def get_model(model_name: str, input_dim: int, hidden_dim: int, output_dim: int, device: torch.device) -> nn.Module:
    key = model_name.lower()
    if key == "dh":
        return Model_DH(input_dim, hidden_dim, output_dim, device).to(device)
    if key == "cp":
        return Model_CP(input_dim, hidden_dim, output_dim, device).to(device)
    if key == "tc":
        return Model_TC(input_dim, hidden_dim, output_dim, device).to(device)
    if key == "ts":
        return Model_TS(input_dim, hidden_dim, output_dim, device).to(device)
    raise ValueError(f"Unknown model: {model_name}")
