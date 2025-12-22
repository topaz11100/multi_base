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


class FeedForwardSNN(nn.Module):
    def __init__(self, neuron_ctor, in_dim: int, hidden: int, out_dim: int, readout_mode: str, warmup_steps: int):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.neuron1 = neuron_ctor((hidden,))
        self.fc2 = nn.Linear(hidden, hidden)
        self.neuron2 = neuron_ctor((hidden,))
        self.fc3 = nn.Linear(hidden, out_dim)
        self.readout_mode = readout_mode
        self.warmup_steps = warmup_steps

    def reset_state(self):
        for neuron in [self.neuron1, self.neuron2]:
            if hasattr(neuron, "reset_state"):
                neuron.reset_state()
            elif hasattr(neuron, "reset"):
                neuron.reset()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        outputs = []
        for t in range(T):
            h1 = self.fc1(x[:, t, :])
            s1 = self.neuron1(h1)
            h2 = self.fc2(s1)
            s2 = self.neuron2(h2)
            logits = self.fc3(s2)
            outputs.append(logits)
        stacked = torch.stack(outputs, dim=1)
        if self.readout_mode == "sum_softmax":
            valid = stacked[:, self.warmup_steps :, :]
            agg = valid.softmax(dim=-1).sum(dim=1)
        else:
            agg = stacked.sum(dim=1)
        return agg


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


def build_model(model_name: str, args) -> nn.Module:
    model_name = model_name.lower()
    readout_mode = args.readout_mode
    warmup_steps = args.warmup_steps
    if model_name.startswith("dh") and args.readout_mode == "sum_logits":
        readout_mode = "sum_softmax"
    if model_name in {"cp", "tc", "ts"}:
        neuron_cls = {"cp": CP_LIF, "tc": partial(TCLIFNode, detach_reset=False), "ts": partial(TSLIFNode, detach_reset=False)}[
            model_name
        ]
        neuron_args: Dict = {
            "cp": {
                "dt": args.dt,
                "v_threshold": args.threshold,
                "tau_init": args.tc_tau_s_high,
                "soft_reset_init": args.tc_gamma,
                "detach_reset": False,
                "surrogate_beta": args.surrogate_beta,
            },
            "tc": {
                "v_threshold": args.threshold,
                "surrogate_function": None,
                "hard_reset": args.hard_reset,
                "detach_reset": False,
                "decay_factor": torch.tensor([[args.tc_beta1, args.tc_beta2]]),
                "gamma": args.tc_gamma,
            },
            "ts": {
                "v_threshold": args.threshold,
                "surrogate_function": None,
                "hard_reset": args.hard_reset,
                "detach_reset": False,
                "decay_factor": torch.tensor([args.tc_tau_s_low, args.tc_tau_s_high, args.tc_tau_d, args.tc_gamma], dtype=torch.float),
                "gamma": args.tc_gamma,
            },
        }
        ctor = neuron_cls if model_name == "cp" else neuron_cls

        def neuron_factory(shape):
            kwargs = neuron_args[model_name]
            if model_name == "cp":
                return ctor(shape, **kwargs)
            return ctor(**kwargs)

        return FeedForwardSNN(neuron_factory, args.in_dim, args.hidden, args.out_dim, readout_mode, warmup_steps)

    if model_name == "dh_sfnn":
        return DHSFNN(
            args.in_dim,
            args.hidden,
            args.out_dim,
            branch=args.dh_branch_sfnn,
            low_n=args.dh_low_n,
            high_n=args.dh_high_n,
            vth=args.dh_vth,
            dt=args.dh_dt,
            readout_mode=readout_mode,
            warmup_steps=warmup_steps,
            layers=args.dh_sfnn_layers,
            device=args.device,
        )
    if model_name == "dh_srnn":
        return DHSRNN(
            args.in_dim,
            args.hidden,
            args.out_dim,
            branch=args.dh_branch_srnn,
            low_n=args.dh_low_n,
            high_n=args.dh_high_n,
            vth=args.dh_vth,
            dt=args.dh_dt,
            readout_mode=readout_mode,
            warmup_steps=warmup_steps,
            device=args.device,
        )
    raise ValueError(f"Unknown model {model_name}")
