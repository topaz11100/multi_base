from __future__ import annotations

from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.common.surrogate import SpikeFn


class TSLIFDenseLayer(nn.Module):
    """
    Temporal-Segment LIF (TS-LIF) dense layer, implemented from the TS-LIF paper (Eq. 5-6).

    Dendrite:
      vd[t] = α1*vd[t-1] + β1*vs[t-1] + (1-α1)*c[t] - γ1*sd[t-1]
      sd[t] = H(vd[t]-v_th)

    Soma:
      vs[t] = α2*vs[t-1] + β2*vd[t] + (1-α2)*c[t] - γ2*ss[t-1]
      ss[t] = H(vs[t]-v_th)

    Mixed output:
      smix[t] = κ*sd[t] + (1-κ)*ss[t]
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        v_th: float = 1.0,
        bias: bool = True,
        spike_fn: Optional[SpikeFn] = None,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.v_th = float(v_th)
        self.spike_fn = spike_fn or SpikeFn(name="mg", lens=0.5, gamma=0.5)

        self.fc = nn.Linear(self.input_dim, self.output_dim, bias=bias)

        # Learnable parameters (per neuron)
        self.alpha1_raw = nn.Parameter(torch.zeros(self.output_dim))
        self.alpha2_raw = nn.Parameter(torch.zeros(self.output_dim))
        self.beta1_raw = nn.Parameter(torch.zeros(self.output_dim))
        self.beta2_raw = nn.Parameter(torch.zeros(self.output_dim))
        self.gamma1_raw = nn.Parameter(torch.zeros(self.output_dim))
        self.gamma2_raw = nn.Parameter(torch.zeros(self.output_dim))
        self.kappa_raw = nn.Parameter(torch.zeros(self.output_dim))

        self.vd: Optional[torch.Tensor] = None
        self.vs: Optional[torch.Tensor] = None
        self.sd_prev: Optional[torch.Tensor] = None
        self.ss_prev: Optional[torch.Tensor] = None

    def alpha1(self) -> torch.Tensor:
        return torch.sigmoid(self.alpha1_raw)

    def alpha2(self) -> torch.Tensor:
        return torch.sigmoid(self.alpha2_raw)

    def beta1(self) -> torch.Tensor:
        return torch.tanh(self.beta1_raw)

    def beta2(self) -> torch.Tensor:
        return torch.tanh(self.beta2_raw)

    def gamma1(self) -> torch.Tensor:
        return F.softplus(self.gamma1_raw)

    def gamma2(self) -> torch.Tensor:
        return F.softplus(self.gamma2_raw)

    def kappa(self) -> torch.Tensor:
        return torch.sigmoid(self.kappa_raw)

    def reset_state(self, batch_size: int, device: torch.device, dtype: torch.dtype = torch.float32) -> None:
        self.vd = torch.zeros(batch_size, self.output_dim, device=device, dtype=dtype)
        self.vs = torch.zeros(batch_size, self.output_dim, device=device, dtype=dtype)
        self.sd_prev = torch.zeros(batch_size, self.output_dim, device=device, dtype=dtype)
        self.ss_prev = torch.zeros(batch_size, self.output_dim, device=device, dtype=dtype)

    def forward_step(self, x_t: torch.Tensor, record: bool = False):
        if self.vd is None or self.vs is None or self.sd_prev is None or self.ss_prev is None:
            self.reset_state(x_t.shape[0], x_t.device, x_t.dtype)

        c_t = self.fc(x_t)

        a1 = self.alpha1().unsqueeze(0)
        a2 = self.alpha2().unsqueeze(0)
        b1 = self.beta1().unsqueeze(0)
        b2 = self.beta2().unsqueeze(0)
        g1 = self.gamma1().unsqueeze(0)
        g2 = self.gamma2().unsqueeze(0)
        k = self.kappa().unsqueeze(0)

        self.vd = a1 * self.vd + b1 * self.vs + (1.0 - a1) * c_t - g1 * self.sd_prev
        sd = self.spike_fn(self.vd - self.v_th)

        self.vs = a2 * self.vs + b2 * self.vd + (1.0 - a2) * c_t - g2 * self.ss_prev
        ss = self.spike_fn(self.vs - self.v_th)

        out = k * sd + (1.0 - k) * ss

        self.sd_prev = sd
        self.ss_prev = ss

        if not record:
            return out

        signals = {
            "dendrite_input": c_t,
            "dendrite_state": self.vd,
            "soma_input": self.vd,
            "soma_state": self.vs,
            "output": out,
        }
        return out, signals

    def forward_sequence(self, x_seq: torch.Tensor, record: bool | Sequence[str] = False):
        # x_seq: (B,T,input_dim)
        B, T, _ = x_seq.shape
        self.reset_state(B, x_seq.device, x_seq.dtype)

        # record can be:
        #   - False: no recording
        #   - True:  record all signals
        #   - Sequence[str]: record only selected keys (e.g., ("soma_state",))
        if record is False:
            record_keys = None
        elif record is True:
            record_keys = ("dendrite_input", "dendrite_state", "soma_input", "soma_state", "output")
        else:
            record_keys = tuple(record)

        # No recording: just return stacked outputs.
        if record_keys is None:
            out_list = []
            for t in range(T):
                out_list.append(self.forward_step(x_seq[:, t], record=False))
            return torch.stack(out_list, dim=1)

        # Recording path: stack both outputs and requested signals.
        out_list = []
        rec_lists: Dict[str, list[torch.Tensor]] = {k: [] for k in record_keys}

        for t in range(T):
            y, sig = self.forward_step(x_seq[:, t], record=True)
            out_list.append(y)

            for k in record_keys:
                if k not in sig:
                    raise KeyError(f"Unknown record key: {k!r}")
                rec_lists[k].append(sig[k])

        out_seq = torch.stack(out_list, dim=1)
        rec = {k: torch.stack(rec_lists[k], dim=1) for k in record_keys}
        return out_seq, rec
    def get_timing_params(self) -> Dict[str, torch.Tensor]:
        return {
            "alpha1": self.alpha1().detach().cpu(),
            "alpha2": self.alpha2().detach().cpu(),
            "kappa": self.kappa().detach().cpu(),
        }

    def active_param_count(self) -> int:
        return sum(int(p.numel()) for p in self.parameters())
