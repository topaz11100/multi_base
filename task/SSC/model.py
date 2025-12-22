from functools import partial
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from neurons.CP_LIF_neuron import CP_LIF
from neurons.TC_LIF_neuron import TCLIFNode
from neurons.TS_LIF_neuron import TSLIFNode
from neurons.DH_SNN_neuron import (
    readout_integrator_test,
    spike_dense_test_denri_wotanh_R,
    spike_rnn_test_denri_wotanh_R,
)


class FeedForwardBackbone(nn.Module):
    def __init__(self, neuron_ctor, in_dim: int, hidden: int, out_dim: int, readout_mode: str = "sum"):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.neuron1 = neuron_ctor((hidden,))
        self.fc2 = nn.Linear(hidden, hidden)
        self.neuron2 = neuron_ctor((hidden,))
        self.fc3 = nn.Linear(hidden, out_dim)
        self.readout_mode = readout_mode

    def reset_state(self):
        for neuron in [self.neuron1, self.neuron2]:
            if hasattr(neuron, "reset_state"):
                neuron.reset_state()
            elif hasattr(neuron, "reset"):
                neuron.reset()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, seq_length, _ = x.shape
        logits_per_t = []
        for t in range(seq_length):
            h1 = self.fc1(x[:, t, :])
            s1 = self.neuron1(h1)
            h2 = self.fc2(s1)
            s2 = self.neuron2(h2)
            logits = self.fc3(s2)
            logits_per_t.append(logits)
        stacked = torch.stack(logits_per_t, dim=1)
        if self.readout_mode == "mean":
            return stacked.mean(dim=1)
        return stacked.sum(dim=1)


class RNNBackbone(nn.Module):
    def __init__(self, neuron_ctor, in_dim: int, hidden: int, out_dim: int, readout_mode: str = "sum"):
        super().__init__()
        self.fc_in = nn.Linear(in_dim, hidden)
        self.neuron = neuron_ctor((hidden,))
        self.readout = nn.Linear(hidden, out_dim)
        self.readout_mode = readout_mode

    def reset_state(self):
        if hasattr(self.neuron, "reset_state"):
            self.neuron.reset_state()
        elif hasattr(self.neuron, "reset"):
            self.neuron.reset()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, seq_length, _ = x.shape
        logits_per_t = []
        for t in range(seq_length):
            h = self.fc_in(x[:, t, :])
            s = self.neuron(h)
            logits_per_t.append(self.readout(s))
        stacked = torch.stack(logits_per_t, dim=1)
        if self.readout_mode == "mean":
            return stacked.mean(dim=1)
        return stacked.sum(dim=1)


class DHSFNN(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden: int,
        out_dim: int,
        branch: int,
        low_n: float,
        high_n: float,
        vth: float,
        dt: float,
        readout_mode: str,
        warmup_steps: int,
        layers: int = 2,
        device: torch.device | None = None,
    ):
        super().__init__()
        device = device or torch.device("cpu")
        self.layers = layers
        self.readout_mode = readout_mode
        self.warmup_steps = warmup_steps
        self.layer1 = spike_dense_test_denri_wotanh_R(
            in_dim, hidden, tau_ninitializer="uniform", low_n=low_n, high_n=high_n, vth=vth, dt=dt, branch=branch, device=device
        )
        self.layer2 = None
        if layers == 2:
            self.layer2 = spike_dense_test_denri_wotanh_R(
                hidden, hidden, tau_ninitializer="uniform", low_n=low_n, high_n=high_n, vth=vth, dt=dt, branch=branch, device=device
            )
        self.readout = readout_integrator_test(hidden, out_dim, dt=dt, device=device)

    def apply_masks(self):
        self.layer1.apply_mask()
        if self.layer2 is not None:
            self.layer2.apply_mask()

    def reset_state(self, batch_size: int):
        self.layer1.set_neuron_state(batch_size)
        if self.layer2 is not None:
            self.layer2.set_neuron_state(batch_size)
        self.readout.set_neuron_state(batch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, seq_length, _ = x.shape
        outputs = []
        for i in range(seq_length):
            _, spike1 = self.layer1(x[:, i, :])
            spike_mid = spike1
            if self.layer2 is not None:
                _, spike_mid = self.layer2(spike1)
            mem_out = self.readout(spike_mid)
            if self.readout_mode == "sum_softmax":
                if i <= self.warmup_steps:
                    continue
                outputs.append(F.softmax(mem_out, dim=1))
            else:
                outputs.append(mem_out)
        stacked = torch.stack(outputs, dim=1) if outputs else torch.zeros(b, 1, mem_out.shape[1], device=mem_out.device)
        return stacked.sum(dim=1)


class DHSRNN(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden: int,
        out_dim: int,
        branch: int,
        low_n: float,
        high_n: float,
        vth: float,
        dt: float,
        readout_mode: str,
        warmup_steps: int,
        device: torch.device | None = None,
    ):
        super().__init__()
        device = device or torch.device("cpu")
        self.readout_mode = readout_mode
        self.warmup_steps = warmup_steps
        self.rnn = spike_rnn_test_denri_wotanh_R(
            in_dim, hidden, tau_ninitializer="uniform", low_n=low_n, high_n=high_n, vth=vth, dt=dt, branch=branch, device=device
        )
        self.readout = readout_integrator_test(hidden, out_dim, dt=dt, device=device)

    def apply_masks(self):
        self.rnn.apply_mask()

    def reset_state(self, batch_size: int):
        self.rnn.set_neuron_state(batch_size)
        self.readout.set_neuron_state(batch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, seq_length, _ = x.shape
        outputs = []
        for i in range(seq_length):
            _, spike1 = self.rnn(x[:, i, :])
            mem_out = self.readout(spike1)
            if self.readout_mode == "sum_softmax":
                if i <= self.warmup_steps:
                    continue
                outputs.append(F.softmax(mem_out, dim=1))
            else:
                outputs.append(mem_out)
        stacked = torch.stack(outputs, dim=1) if outputs else torch.zeros(b, 1, mem_out.shape[1], device=mem_out.device)
        return stacked.sum(dim=1)


def build_model(neuron_type: str, args) -> nn.Module:
    neuron_type = neuron_type.lower()
    readout_mode = args.readout_mode
    warmup_steps = args.warmup_steps
    hidden = args.hidden
    in_dim = args.nb_units
    out_dim = args.num_classes

    if neuron_type in {"cp", "tc", "ts"}:
        neuron_cls = {"cp": CP_LIF, "tc": partial(TCLIFNode, detach_reset=False), "ts": partial(TSLIFNode, detach_reset=False)}[
            neuron_type
        ]
        neuron_args: Dict = {
            "cp": {
                "dt": args.dt,
                "v_threshold": args.vth,
                "tau_init": args.tau_init,
                "soft_reset_init": args.soft_reset,
                "detach_reset": False,
                "surrogate_beta": args.sg_beta,
            },
            "tc": {
                "v_threshold": args.vth,
                "surrogate_function": None,
                "hard_reset": args.reset_mode == "hard",
                "detach_reset": False,
                "decay_factor": torch.tensor([[args.beta1, args.beta2]]),
                "gamma": args.gamma,
            },
            "ts": {
                "v_threshold": args.vth,
                "surrogate_function": None,
                "hard_reset": args.reset_mode == "hard",
                "detach_reset": False,
                "decay_factor": torch.tensor([args.tau_s_low, args.tau_s_high, args.tau_d, args.gamma], dtype=torch.float),
                "gamma": args.gamma,
            },
        }
        backbone_cls = FeedForwardBackbone if args.backbone == "ff" else RNNBackbone
        return backbone_cls(partial(neuron_cls, **neuron_args[neuron_type]), in_dim, hidden, out_dim, readout_mode=args.readout_mode)

    if neuron_type == "dh-sfnn":
        return DHSFNN(
            in_dim=in_dim,
            hidden=hidden,
            out_dim=out_dim,
            branch=args.dh_branch_sfnn,
            low_n=args.dh_low_n,
            high_n=args.dh_high_n,
            vth=args.vth,
            dt=args.dt,
            readout_mode=readout_mode,
            warmup_steps=warmup_steps,
            layers=args.dh_sfnn_layers,
            device=torch.device(args.device),
        )
    if neuron_type == "dh-srnn":
        return DHSRNN(
            in_dim=in_dim,
            hidden=hidden,
            out_dim=out_dim,
            branch=args.dh_branch_srnn,
            low_n=args.dh_low_n,
            high_n=args.dh_high_n,
            vth=args.vth,
            dt=args.dt,
            readout_mode=readout_mode,
            warmup_steps=warmup_steps,
            device=torch.device(args.device),
        )
    raise ValueError(f"Unknown neuron type: {neuron_type}")
