"""Model factory wrapping the four neuron variants for the multiscale XOR task."""
from __future__ import annotations

from typing import Callable, Dict

import torch
import torch.nn as nn

from neurons.CP_LIF_neuron import CP_LIF
from neurons.DH_SNN_neuron import readout_integrator_test, spike_dense_test_denri_wotanh_R_xor
from neurons.TC_LIF_neuron import TCLIFNode
from neurons.TS_LIF_neuron import TSLIFNode


class _TimeLoopModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, device: torch.device):
        super().__init__()
        self.device = device
        self.fc_in = nn.Linear(input_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def reset_state(self, batch_size: int) -> None:  # pragma: no cover - implemented by subclasses
        raise NotImplementedError

    def neuron_forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, time_steps, _ = x.shape
        outputs = []
        self.reset_state(batch_size)
        for t in range(time_steps):
            h = self.fc_in(x[:, t, :])
            s = self.neuron_forward(h)
            outputs.append(self.fc_out(s).unsqueeze(1))
        return torch.cat(outputs, dim=1)


class DHModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, device: torch.device, **neuron_kwargs):
        super().__init__()
        self.device = device
        self.dense_1 = spike_dense_test_denri_wotanh_R_xor(
            input_dim,
            hidden_dim,
            tau_minitializer=neuron_kwargs.get("tau_initializer", "uniform"),
            low_m=neuron_kwargs.get("low_m", 0),
            high_m=neuron_kwargs.get("high_m", 4),
            branch=neuron_kwargs.get("branch", 1),
            vth=neuron_kwargs.get("vth", 1),
            dt=neuron_kwargs.get("dt", 1),
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


class CPModel(_TimeLoopModel):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, device: torch.device, **neuron_kwargs):
        super().__init__(input_dim, hidden_dim, output_dim, device)
        self.lif = CP_LIF((hidden_dim,), **neuron_kwargs)

    def reset_state(self, batch_size: int) -> None:
        self.lif.reset_state()

    def neuron_forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lif(x)


class TCModel(_TimeLoopModel):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, device: torch.device, **neuron_kwargs):
        super().__init__(input_dim, hidden_dim, output_dim, device)
        self.node = TCLIFNode(step_mode="s", **neuron_kwargs)

    def reset_state(self, batch_size: int) -> None:
        self.node.reset()

    def neuron_forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.node(x)


class TSModel(_TimeLoopModel):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, device: torch.device, **neuron_kwargs):
        super().__init__(input_dim, hidden_dim, output_dim, device)
        # Recreate the TS neuron with configurable dimensions.
        self.node = TSLIFNode(step_mode="s", **neuron_kwargs)

    def reset_state(self, batch_size: int) -> None:
        self.node.reset()

    def neuron_forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.node(x)


def build_model(neuron_name: str, input_dim: int, hidden_dim: int, output_dim: int, device: torch.device, neuron_kwargs: Dict) -> nn.Module:
    key = neuron_name.lower()
    builders: Dict[str, Callable[..., nn.Module]] = {
        "dh": lambda: DHModel(input_dim, hidden_dim, output_dim, device, **neuron_kwargs),
        "cp": lambda: CPModel(input_dim, hidden_dim, output_dim, device, **neuron_kwargs),
        "tc": lambda: TCModel(input_dim, hidden_dim, output_dim, device, **neuron_kwargs),
        "ts": lambda: TSModel(input_dim, hidden_dim, output_dim, device, **neuron_kwargs),
    }
    if key not in builders:
        raise ValueError(f"Unknown neuron type: {neuron_name}")
    model = builders[key]().to(device)

    if key == "dh":
        # Ensure the DH branch connectivity mask is applied before any training steps
        model.dense_1.apply_mask()

        # Register a single gradient hook to zero gradients outside the mask
        if not getattr(model.dense_1, "_dh_mask_hooked", False):
            def _mask_grad_hook(grad: torch.Tensor) -> torch.Tensor:
                return grad * model.dense_1.mask

            model.dense_1.dense.weight.register_hook(_mask_grad_hook)
            model.dense_1._dh_mask_hooked = True

    return model
