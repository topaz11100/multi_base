from __future__ import annotations

from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn

from src.common.surrogate import SpikeFn


class TCLIFDenseLayer(nn.Module):
    """
    Two-Compartment LIF (TC-LIF) dense layer, adapted from TC-LIF author implementation.
    State:
      - v1: dendritic compartment
      - v2: somatic compartment (used for spiking)
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        v_th: float = 1.0,
        gamma: float = 0.5,
        bias: bool = True,
        spike_fn: Optional[SpikeFn] = None,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.v_th = float(v_th)
        self.gamma = float(gamma)
        self.spike_fn = spike_fn or SpikeFn(name="mg", lens=0.5, gamma=0.5)

        self.fc = nn.Linear(self.input_dim, self.output_dim, bias=bias)

        # decay_factor[:,0] for coupling term of v2 into v1; decay_factor[:,1] for coupling term of v1 into v2.
        self.decay_factor = nn.Parameter(torch.zeros(self.output_dim, 2))

        self.v1: Optional[torch.Tensor] = None
        self.v2: Optional[torch.Tensor] = None
        self.spk: Optional[torch.Tensor] = None

    def reset_state(self, batch_size: int, device: torch.device, dtype: torch.dtype = torch.float32) -> None:
        self.v1 = torch.zeros(batch_size, self.output_dim, device=device, dtype=dtype)
        self.v2 = torch.zeros(batch_size, self.output_dim, device=device, dtype=dtype)
        self.spk = torch.zeros(batch_size, self.output_dim, device=device, dtype=dtype)

    def forward_step(self, x_t: torch.Tensor, record: bool = False):
        if self.v1 is None or self.v2 is None or self.spk is None:
            self.reset_state(x_t.shape[0], x_t.device, x_t.dtype)

        i_t = self.fc(x_t)
        df = torch.sigmoid(self.decay_factor).unsqueeze(0)  # (1,N,2)

        # Update (following author code semantics)
        self.v1 = self.v1 - df[..., 0] * self.v2 + i_t
        self.v2 = self.v2 + df[..., 1] * self.v1

        # IMPORTANT: record pre-reset membrane for membrane-potential readout.
        v1_pre = self.v1
        v2_pre = self.v2

        spk = self.spike_fn(v2_pre - self.v_th)

        # Soft reset
        self.v1 = self.v1 - spk * self.gamma
        self.v2 = self.v2 - spk * self.v_th
        self.spk = spk

        if not record:
            return spk

        signals = {
            "dendrite_input": i_t,
            "dendrite_state": v1_pre,
            "soma_input": v2_pre,
            "soma_state": v2_pre,
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
        df = torch.sigmoid(self.decay_factor.detach().cpu())
        return {"decay_factor_0": df[:, 0], "decay_factor_1": df[:, 1]}

    def active_param_count(self) -> int:
        return sum(int(p.numel()) for p in self.parameters())
