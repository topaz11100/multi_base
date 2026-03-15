from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.common.surrogate import SpikeFn


class LIFDenseLayer(nn.Module):
    """
    Baseline LIF dense spiking layer.
    - Synapse: Linear(input_dim -> output_dim)
    - Neuron: LIF with fixed decay alpha, soft reset via subtracting v_th * spike_prev
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        alpha: float = 0.9,
        v_th: float = 1.0,
        bias: bool = True,
        spike_fn: Optional[SpikeFn] = None,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)

        self.alpha = float(alpha)
        self.v_th = float(v_th)
        self.spike_fn = spike_fn or SpikeFn(name="mg", lens=0.5, gamma=0.5)

        self.fc = nn.Linear(self.input_dim, self.output_dim, bias=bias)

        self.mem: Optional[torch.Tensor] = None
        self.spk: Optional[torch.Tensor] = None

    def reset_state(self, batch_size: int, device: torch.device, dtype: torch.dtype = torch.float32) -> None:
        self.mem = torch.zeros(batch_size, self.output_dim, device=device, dtype=dtype)
        self.spk = torch.zeros(batch_size, self.output_dim, device=device, dtype=dtype)

    def forward_step(self, x_t: torch.Tensor, record: bool = False):
        if self.mem is None or self.spk is None:
            self.reset_state(x_t.shape[0], x_t.device, x_t.dtype)

        i_t = self.fc(x_t)
        self.mem = self.mem * self.alpha + i_t - self.v_th * self.spk
        spk = self.spike_fn(self.mem - self.v_th)
        self.spk = spk

        if not record:
            return spk

        signals = {
            "dendrite_input": i_t,
            "dendrite_state": self.mem,
            "soma_input": i_t,
            "soma_state": self.mem,
            "output": spk,
        }
        return spk, signals

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
        return {"alpha": torch.tensor([self.alpha])}

    def active_param_count(self) -> int:
        return sum(int(p.numel()) for p in self.parameters())
