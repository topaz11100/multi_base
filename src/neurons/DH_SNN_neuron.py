from __future__ import annotations

from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.common.surrogate import SpikeFn


class DHSNNDenseLayer(nn.Module):
    """
    Baseline DH-SNN dense layer (Temporal Dendritic Heterogeneity), adapted from DH-SNN author code.

    Key properties (baseline):
      - Branch count D is fixed (branch).
      - Sparse / partitioned input connectivity pattern (mask), default non-overlapping partition (1/branch).
      - Branch current: i_d[t] = α_d i_d[t-1] + (1-α_d) I_d[t]
      - Soma membrane: u[t] = β u[t-1] + (1-β) sum_d i_d[t] - v_th * o[t-1]
      - Spike: o[t] = H(u[t] - v_th)
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        branch: int = 4,
        v_th: float = 1.0,
        dt: float = 1.0,
        bias: bool = True,
        test_sparsity: bool = False,
        sparsity: float = 0.5,
        mask_share: int = 1,
        spike_fn: Optional[SpikeFn] = None,
        tau_m_init: float = 0.0,
        tau_n_init: float = 0.0,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.branch = int(branch)
        self.v_th = float(v_th)
        self.dt = float(dt)
        self.test_sparsity = bool(test_sparsity)
        self.sparsity = float(sparsity) if test_sparsity else (1.0 / float(self.branch))
        self.mask_share = int(mask_share)
        self.spike_fn = spike_fn or SpikeFn(name="mg", lens=0.5, gamma=0.5)

        self.pad = ((self.input_dim // self.branch) * self.branch + self.branch - self.input_dim) % self.branch
        in_features = self.input_dim + self.pad
        out_features = self.output_dim * self.branch

        self.fc = nn.Linear(in_features, out_features, bias=bias)

        # Timing parameters
        self.tau_m = nn.Parameter(torch.full((self.output_dim,), float(tau_m_init)))
        self.tau_n = nn.Parameter(torch.full((self.output_dim, self.branch), float(tau_n_init)))

        # Connection mask (buffer)
        mask = self._create_mask(in_features)
        self.register_buffer("mask", mask, persistent=False)

        # States
        self.mem: Optional[torch.Tensor] = None         # (B,N)
        self.spk: Optional[torch.Tensor] = None         # (B,N)
        self.d_state: Optional[torch.Tensor] = None     # (B,N,D)

    def _create_mask(self, input_size: int) -> torch.Tensor:
        mask = torch.zeros(self.output_dim * self.branch, input_size)
        groups = max(1, self.output_dim // self.mask_share)
        for gi in range(groups):
            seq = torch.randperm(input_size)
            for d in range(self.branch):
                if self.test_sparsity:
                    # Potential wrap-around as in author code
                    start = d * input_size // self.branch
                    span = int(input_size * self.sparsity)
                    end = start + span
                    if end > input_size:
                        # NOTE: slice end is exclusive; include the last element.
                        part1 = seq[start:input_size]
                        part2 = seq[: end - input_size]
                        idxs = torch.cat([part1, part2], dim=0)
                    else:
                        idxs = seq[start:end]
                else:
                    start = d * input_size // self.branch
                    end = (d + 1) * input_size // self.branch
                    idxs = seq[start:end]

                for k in range(self.mask_share):
                    n = gi * self.mask_share + k
                    if n >= self.output_dim:
                        continue
                    row = n * self.branch + d
                    mask[row, idxs] = 1.0
        return mask

    def alpha_branch(self) -> torch.Tensor:
        # α_d in (0,1), shape (N,D)
        return torch.sigmoid(self.tau_n)

    def beta_soma(self) -> torch.Tensor:
        # β in (0,1), shape (N,)
        return torch.sigmoid(self.tau_m)

    def reset_state(self, batch_size: int, device: torch.device, dtype: torch.dtype = torch.float32) -> None:
        self.mem = torch.zeros(batch_size, self.output_dim, device=device, dtype=dtype)
        self.spk = torch.zeros(batch_size, self.output_dim, device=device, dtype=dtype)
        self.d_state = torch.zeros(batch_size, self.output_dim, self.branch, device=device, dtype=dtype)

    def forward_step(self, x_t: torch.Tensor, record: bool = False):
        if self.mem is None or self.spk is None or self.d_state is None:
            self.reset_state(x_t.shape[0], x_t.device, x_t.dtype)

        if self.pad > 0:
            padding = torch.zeros(x_t.shape[0], self.pad, device=x_t.device, dtype=x_t.dtype)
            x_in = torch.cat([x_t, padding], dim=1)
        else:
            x_in = x_t

        # Apply mask deterministically in forward
        w_eff = self.fc.weight * self.mask.to(self.fc.weight.dtype)
        d_in = F.linear(x_in, w_eff, self.fc.bias)  # (B, N*D)
        d_in = d_in.view(-1, self.output_dim, self.branch)  # (B,N,D)

        alpha = self.alpha_branch().unsqueeze(0)  # (1,N,D)
        self.d_state = alpha * self.d_state + (1.0 - alpha) * d_in

        soma_in = self.d_state.sum(dim=2)  # (B,N)

        beta = self.beta_soma().unsqueeze(0)  # (1,N)
        self.mem = self.mem * beta + (1.0 - beta) * soma_in - self.v_th * self.spk
        spk = self.spike_fn(self.mem - self.v_th)
        self.spk = spk

        if not record:
            return spk

        signals = {
            "dendrite_input": d_in,
            "dendrite_state": self.d_state,
            "soma_input": soma_in,
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
        return {
            "alpha": self.alpha_branch().detach().cpu().flatten(),
            "beta": self.beta_soma().detach().cpu().flatten(),
        }

    def active_param_count(self) -> int:
        # Fixed branch baseline => all params active
        return sum(int(p.numel()) for p in self.parameters())
