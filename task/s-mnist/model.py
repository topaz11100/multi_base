from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

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
from neurons import surrogate  # [유지] surrogate 모듈 임포트


@dataclass
class ModelConfig:
    in_dim: int
    hidden_dims: List[int]
    out_dim: int = 10
    arch: str = "ff"  # ff or fb
    readout_mode: str = "sum"
    warmup: int = 0


class FeedForwardNet(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: List[int], out_dim: int, neuron_ctor, readout_mode: str = "sum"):
        super().__init__()
        layers: List[nn.Module] = []
        prev_dim = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(neuron_ctor((h,)))
            prev_dim = h
        self.feature = nn.Sequential(*layers)
        self.readout = nn.Linear(prev_dim, out_dim)
        self.readout_mode = readout_mode

    # [유지] RecursionError 방지 코드
    def reset_state(self):
        for m in self.modules():
            if m is self:
                continue
            if hasattr(m, "reset_state"):
                try:
                    m.reset_state()
                except TypeError:
                    pass
            elif hasattr(m, "reset"):
                try:
                    m.reset()
                except TypeError:
                    pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = []
        for t in range(x.size(1)):
            h = self.feature(x[:, t, :])
            out = self.readout(h)
            outputs.append(out)
        stacked = torch.stack(outputs, dim=1)
        if self.readout_mode == "mean":
            return stacked.mean(dim=1)
        return stacked.sum(dim=1)


class SimpleSRNN(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: List[int], out_dim: int, neuron_ctor, readout_mode: str = "sum"):
        super().__init__()
        self.in_to_h1 = nn.Linear(in_dim, hidden_dims[0])
        self.h1_to_h1 = nn.Linear(hidden_dims[0], hidden_dims[0])
        self.h2 = None
        if len(hidden_dims) > 1:
            self.h1_to_h2 = nn.Linear(hidden_dims[0], hidden_dims[1])
            self.h2_to_h2 = nn.Linear(hidden_dims[1], hidden_dims[1])
            self.h2 = True
            h_last = hidden_dims[1]
        else:
            h_last = hidden_dims[0]
        self.out = nn.Linear(h_last, out_dim)
        self.neuron = neuron_ctor((hidden_dims[0],))
        self.neuron2 = neuron_ctor((hidden_dims[1],)) if self.h2 else None
        self.readout_mode = readout_mode

    def reset_state(self, batch_size: int | None = None, device: torch.device | None = None):
        for m in [self.neuron, self.neuron2]:
            if m is None:
                continue
            if hasattr(m, "reset_state"):
                try:
                    m.reset_state()
                except TypeError:
                    if batch_size is not None:
                        m.reset_state(batch_size)
            elif hasattr(m, "reset"):
                m.reset()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        h1_spike = torch.zeros(B, self.in_to_h1.out_features, device=x.device)
        h2_spike = None
        if self.h2:
            h2_spike = torch.zeros(B, self.h1_to_h2.out_features, device=x.device)
        outputs = []
        for t in range(x.size(1)):
            input_t = x[:, t, :]
            h1_input = self.in_to_h1(input_t) + self.h1_to_h1(h1_spike)
            h1_spike = self.neuron(h1_input)
            if self.h2:
                h2_input = self.h1_to_h2(h1_spike) + self.h2_to_h2(h2_spike)
                h2_spike = self.neuron2(h2_input)
                out_t = self.out(h2_spike)
            else:
                out_t = self.out(h1_spike)
            outputs.append(out_t)
        stacked = torch.stack(outputs, dim=1)
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
        warmup: int,
        layers: int = 2,
        device: torch.device | None = None,
    ):
        super().__init__()
        device = device or torch.device("cpu")
        self.readout_mode = readout_mode
        self.warmup = warmup
        self.layers = layers
        self.layer1 = spike_dense_test_denri_wotanh_R(
            in_dim, hidden, tau_ninitializer="uniform", low_n=low_n, high_n=high_n, vth=vth, dt=dt, branch=branch, device=device
        )
        self.layer2 = None
        if layers > 1:
            self.layer2 = spike_dense_test_denri_wotanh_R(
                hidden, hidden, tau_ninitializer="uniform", low_n=low_n, high_n=high_n, vth=vth, dt=dt, branch=branch, device=device
            )
        self.readout = readout_integrator_test(hidden, out_dim, dt=dt, device=device)

    def apply_masks(self):
        self.layer1.apply_mask()
        if self.layer2 is not None:
            self.layer2.apply_mask()

    def reset_state(self, batch_size: int, device: torch.device | None = None):
        dev = device or next(self.parameters()).device
        self.layer1.set_neuron_state(batch_size, device=dev)
        if self.layer2 is not None:
            self.layer2.set_neuron_state(batch_size, device=dev)
        self.readout.set_neuron_state(batch_size, device=dev)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = []
        for t in range(x.size(1)):
            _, spike1 = self.layer1(x[:, t, :])
            spike_mid = spike1
            if self.layer2 is not None:
                _, spike_mid = self.layer2(spike1)
            mem = self.readout(spike_mid)
            if self.readout_mode == "sum_softmax":
                if t <= self.warmup:
                    continue
                outputs.append(F.softmax(mem, dim=1))
            else:
                outputs.append(mem)
        stacked = torch.stack(outputs, dim=1) if outputs else torch.zeros(x.size(0), 1, self.readout.out_features, device=x.device)
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
        warmup: int,
        device: torch.device | None = None,
    ):
        super().__init__()
        device = device or torch.device("cpu")
        self.readout_mode = readout_mode
        self.warmup = warmup
        self.rnn = spike_rnn_test_denri_wotanh_R(
            in_dim, hidden, tau_ninitializer="uniform", low_n=low_n, high_n=high_n, vth=vth, dt=dt, branch=branch, device=device
        )
        self.readout = readout_integrator_test(hidden, out_dim, dt=dt, device=device)

    def apply_masks(self):
        self.rnn.apply_mask()

    def reset_state(self, batch_size: int, device: torch.device | None = None):
        dev = device or next(self.parameters()).device
        self.rnn.set_neuron_state(batch_size, device=dev)
        self.readout.set_neuron_state(batch_size, device=dev)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = []
        for t in range(x.size(1)):
            _, spike1 = self.rnn(x[:, t, :])
            mem = self.readout(spike1)
            if self.readout_mode == "sum_softmax":
                if t <= self.warmup:
                    continue
                outputs.append(F.softmax(mem, dim=1))
            else:
                outputs.append(mem)
        stacked = torch.stack(outputs, dim=1) if outputs else torch.zeros(x.size(0), 1, self.readout.out_features, device=x.device)
        return stacked.sum(dim=1)


def build_model(neuron_name: str, cfg: ModelConfig, neuron_kwargs: Dict) -> nn.Module:
    name = neuron_name.lower()
    readout_mode = cfg.readout_mode
    if name.startswith("dh") and readout_mode == "sum":
        readout_mode = "sum_softmax"

    if cfg.arch != "ff" and len(cfg.hidden_dims) < 2:
        raise ValueError(
            "SRNN architecture requires at least two hidden dimensions (e.g., --hidden 256 256)"
        )

    if name in {"cp", "tc", "ts"}:
        if name == "cp":
            ctor = lambda shape: CP_LIF(shape, **neuron_kwargs.get("cp", {}))
        elif name == "tc":
            # [수정] partial 제거, surrogate.Triangle.apply 직접 할당
            kwargs = neuron_kwargs.get("tc", {}).copy()
            if "surrogate_function" not in kwargs:
                kwargs["surrogate_function"] = surrogate.Triangle.apply
            ctor = lambda shape: TCLIFNode(**kwargs, detach_reset=False)
        else:
            # [수정] partial 제거, surrogate.Triangle.apply 직접 할당
            kwargs = neuron_kwargs.get("ts", {}).copy()
            if "surrogate_function" not in kwargs:
                kwargs["surrogate_function"] = surrogate.Triangle.apply
            ctor = lambda shape: TSLIFNode(**kwargs, detach_reset=False, neuron_shape=shape)
            
        if cfg.arch == "ff":
            return FeedForwardNet(cfg.in_dim, cfg.hidden_dims, cfg.out_dim, ctor, readout_mode)
        return SimpleSRNN(cfg.in_dim, cfg.hidden_dims, cfg.out_dim, ctor, readout_mode)

    if name == "dh-sfnn":
        dh_kwargs = neuron_kwargs.get("dh-sfnn", {})
        branch = dh_kwargs.get("branch", 4)
        low_n = dh_kwargs.get("low_n", 2)
        high_n = dh_kwargs.get("high_n", 6)
        vth = dh_kwargs.get("vth", 1.0)
        dt = dh_kwargs.get("dt", 1.0)
        layers = dh_kwargs.get("layers", 2)
        return DHSFNN(
            cfg.in_dim,
            cfg.hidden_dims[0],
            cfg.out_dim,
            branch=branch,
            low_n=low_n,
            high_n=high_n,
            vth=vth,
            dt=dt,
            readout_mode=readout_mode,
            warmup=cfg.warmup,
            layers=layers,
        )

    if name == "dh-srnn":
        dh_kwargs = neuron_kwargs.get("dh-srnn", {})
        branch = dh_kwargs.get("branch", 8)
        low_n = dh_kwargs.get("low_n", 2)
        high_n = dh_kwargs.get("high_n", 6)
        vth = dh_kwargs.get("vth", 1.0)
        dt = dh_kwargs.get("dt", 1.0)
        return DHSRNN(
            cfg.in_dim,
            cfg.hidden_dims[0],
            cfg.out_dim,
            branch=branch,
            low_n=low_n,
            high_n=high_n,
            vth=vth,
            dt=dt,
            readout_mode=readout_mode,
            warmup=cfg.warmup,
        )

    raise ValueError(f"Unknown neuron {neuron_name}")