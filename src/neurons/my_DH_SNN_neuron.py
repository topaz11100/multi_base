from __future__ import annotations

import math
from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.common.surrogate import SpikeFn


def _ema_orthogonality_loss(alpha: torch.Tensor) -> torch.Tensor:
    """Closed-form squared cosine similarity for EMA kernels (varidble_dendric.md).

    h_d[n] = (1-a_d) a_d^n, 0<a_d<1

    cos(h_i, h_j) = sqrt((1-a_i^2)(1-a_j^2)) / (1 - a_i a_j)

    We return sum_{i<j} cos^2.
    alpha: (D,) in (0,1)
    """

    D = int(alpha.numel())
    if D <= 1:
        return alpha.new_zeros(())
    a = alpha.view(-1)
    ai = a.view(D, 1)
    aj = a.view(1, D)
    num = (1.0 - ai * ai) * (1.0 - aj * aj)
    den = (1.0 - ai * aj).clamp_min(1e-6) ** 2
    cos2 = num / den
    return torch.triu(cos2, diagonal=1).sum()


def _soft_branch_mask_from_s(s: torch.Tensor, branch: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (mask, D_int) for soft dendrite masking.

    Mask rule (0-indexed in code, 1-indexed in docs):
      - for d < floor(s): mask=1
      - for d = floor(s): mask = s - floor(s)
      - else: mask=0

    Returns:
      mask:  (N,D) in [0,1]
      D_int: (N,) number of branches with mask>0 (ceil(s) except when s is integer).
    """
    if s.ndim != 1:
        raise ValueError(f"s must be 1D (got shape={tuple(s.shape)})")
    D = int(branch)
    s_c = s.clamp_min(0.0).clamp_max(float(D))
    k = torch.floor(s_c).to(torch.int64)
    frac = (s_c - k.to(s_c.dtype)).clamp(0.0, 1.0)
    idx = torch.arange(D, device=s.device, dtype=torch.int64).view(1, D)
    k2 = k.view(-1, 1)
    mask = (idx < k2).to(s.dtype)
    mask = mask + (idx == k2).to(s.dtype) * frac.view(-1, 1) * (k2 < D).to(s.dtype)
    D_int = (k + (frac > 0).to(torch.int64)).clamp(min=1, max=D)
    return mask, D_int


def _hard_branch_mask_from_s(s: torch.Tensor, branch: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (mask_hard, D_hard) for hard dendrite masking.

    Hard branch count rule (varidble_dendric.md):
      - \tilde{s}=min(D,s)
      - D_hard=floor(\tilde{s}+1/2) (round-half-up)

    Returns:
      mask_hard: (N,D) in {0,1}
      D_hard:    (N,) integer number of active branches.
    """
    if s.ndim != 1:
        raise ValueError(f"s must be 1D (got shape={tuple(s.shape)})")
    D = int(branch)
    s_c = s.clamp_min(0.0).clamp_max(float(D))
    D_hard = torch.floor(s_c + 0.5).to(torch.int64).clamp(min=1, max=D)
    idx = torch.arange(D, device=s.device, dtype=torch.int64).view(1, D)
    mask_hard = (idx < D_hard.view(-1, 1)).to(s.dtype)
    return mask_hard, D_hard


def _ema_orthogonality_loss_weighted(alpha: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Weighted EMA orthogonality loss (mean over neurons)."""
    if alpha.ndim != 2 or w.ndim != 2 or alpha.shape != w.shape:
        raise ValueError(f"alpha and w must be (N,D) with same shape (got {alpha.shape}, {w.shape})")
    N, D = alpha.shape
    if D <= 1:
        return alpha.new_zeros(())
    ai = alpha.unsqueeze(2)
    aj = alpha.unsqueeze(1)
    num = (1.0 - ai * ai) * (1.0 - aj * aj)
    den = (1.0 - ai * aj).clamp_min(1e-6) ** 2
    cos2 = (num / den) * (w.unsqueeze(2) * w.unsqueeze(1))
    return torch.triu(cos2, diagonal=1).sum(dim=(1, 2)).mean()


class MyDHSNNDenseLayer(nn.Module):
    """Proposed DH-SNN dense spiking layer (paper/proposed/my_DH_SNN_neuron.md).

    - Dense branch weights (no sparse routing mask)
    - Variable dendrite count parameter s (continuous), with soft/hard masking
    - Branch tensor size is fixed by `branch` for GPU-friendly shapes
    - Continuous s is constrained only by (S_min, S_max)
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        branch: int = 8,
        S_min: float = 1.0,
        S_max: Optional[float] = None,
        v_th: float = 1.0,
        bias: bool = True,
        spike_fn: Optional[SpikeFn] = None,
        tau_m_init: float = 0.0,
        tau_n_init: float = 0.0,
        s_init: Optional[float] = None,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.branch = int(branch)
        self.v_th = float(v_th)

        if S_max is None:
            S_max = float(self.branch)
        self.S_min = float(S_min)
        self.S_max = float(S_max)

        if not (1 <= self.branch):
            raise ValueError(f"branch must be >= 1 (got {self.branch})")
        if not (1.0 <= self.S_min <= self.S_max):
            raise ValueError(f"Require 1 <= S_min <= S_max (got S_min={self.S_min}, S_max={self.S_max})")
        # NOTE: S_max may exceed `branch` (D). The mask saturates at D via clamp,
        # while soma normalization keeps using 1/s.

        self.spike_fn = spike_fn or SpikeFn(name="mg", lens=0.5, gamma=0.5)

        # Branch weights: treat as one big linear with out_dim = output_dim * branch
        self.W = nn.Parameter(torch.empty(self.output_dim * self.branch, self.input_dim))
        self.bias = nn.Parameter(torch.zeros(self.output_dim * self.branch)) if bias else None
        nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))  # type: ignore

        # Initialization stabilization (varidble_dendric.md §7): W <- sqrt(D) * W
        with torch.no_grad():
            self.W.mul_(math.sqrt(float(self.branch)))

        # Timing factors (raw -> sigmoid)
        self.tau_m = nn.Parameter(torch.full((self.output_dim,), float(tau_m_init)))
        self.tau_n = nn.Parameter(torch.full((self.output_dim, self.branch), float(tau_n_init)))

        # Structure parameter s in [S_min, S_max] (per-neuron) via sigmoid reparam.
        # Default policy: start from the *S_min side* (sparse branches) and let training grow s if needed.
        # NOTE: We still clamp the normalized ratio to [eps, 1-eps] to avoid sigmoid saturation.
        if s_init is None:
            s_init = float(self.S_min)
        s0 = torch.full((self.output_dim,), float(s_init), dtype=torch.float32)
        if abs(self.S_max - self.S_min) < 1e-12:
            s_norm = torch.zeros_like(s0)
        else:
            s_norm = (s0 - float(self.S_min)) / float(self.S_max - self.S_min)
        eps = 1e-2
        s_norm = s_norm.clamp(eps, 1.0 - eps)
        s_raw_init = torch.log(s_norm / (1.0 - s_norm))
        self.s_raw = nn.Parameter(s_raw_init.clone().detach())

        # Two-stage training support (soft mask -> hard mask).
        self.register_buffer("_hard_enabled", torch.zeros((), dtype=torch.uint8), persistent=True)
        self.register_buffer(
            "_s_hard",
            torch.full((self.output_dim,), float(self.S_min), dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer("_ste_enabled", torch.zeros((), dtype=torch.uint8), persistent=False)

        # States
        self.mem: Optional[torch.Tensor] = None  # (B,N)
        self.spk: Optional[torch.Tensor] = None  # (B,N)
        self.d_state: Optional[torch.Tensor] = None  # (B,N,D)

    # ---------------------------------------------------------------------
    # Structure / timing helpers
    # ---------------------------------------------------------------------

    def s(self) -> torch.Tensor:
        # After hardening, s becomes an integer (float tensor) and is frozen.
        if int(self._hard_enabled.item()) == 1:
            return self._s_hard.to(device=self.s_raw.device, dtype=self.s_raw.dtype)
        if abs(self.S_max - self.S_min) < 1e-12:
            return torch.full((self.output_dim,), float(self.S_max), device=self.s_raw.device, dtype=self.s_raw.dtype)
        return self.S_min + (self.S_max - self.S_min) * torch.sigmoid(self.s_raw)

    def d_int(self) -> torch.Tensor:
        """Per-neuron integer number of active branches in the *current* masking mode.

        - hardened: integer s_hard (frozen)
        - STE:      uses D_hard (forward-hard structure)
        - soft:     counts branches with mask>0 (ceil(s) except when s is integer)
        """
        if int(self._hard_enabled.item()) == 1:
            return self._s_hard.to(device=self.s_raw.device, dtype=torch.int64).clamp(min=1, max=int(self.branch))
        if int(self._ste_enabled.item()) == 1:
            _, D_hard = _hard_branch_mask_from_s(self.s().to(torch.float32), self.branch)
            return D_hard
        _, D_int = _soft_branch_mask_from_s(self.s(), self.branch)
        return D_int

    def soft_mask(self, dtype: torch.dtype) -> torch.Tensor:
        """Per-neuron mask M(s) with shape (N,D).

        Stage A (soft): values in [0,1]
        Stage A (STE): forward uses hard mask in {0,1}, backward uses soft gradients
        Stage B (hardened): values in {0,1} with frozen s
        """
        if int(self._hard_enabled.item()) == 1:
            D_int = self.d_int().to(torch.int64)
            idx = torch.arange(int(self.branch), device=D_int.device, dtype=torch.int64).view(1, int(self.branch))
            return (idx < D_int.view(-1, 1)).to(dtype)

        s = self.s().to(dtype)
        mask_soft, _ = _soft_branch_mask_from_s(s, self.branch)

        if int(self._ste_enabled.item()) == 1:
            mask_hard, _ = _hard_branch_mask_from_s(s, self.branch)
            # STE: forward=hard, backward=soft
            return mask_soft + (mask_hard - mask_soft).detach()

        return mask_soft

    @torch.no_grad()
    def enable_ste(self, enabled: bool) -> None:
        """Enable/disable STE mode (forward hard, backward soft) during stage A."""
        if int(self._hard_enabled.item()) == 1:
            self._ste_enabled.fill_(0)
            return
        self._ste_enabled.fill_(1 if bool(enabled) else 0)

    @torch.no_grad()
    def harden_branches(self) -> None:
        """Transition: soft mask -> hard mask, and freeze s.

        Implements varidble_dendric.md:
          s_hard <- floor(min(D, s) + 1/2)  (round-half-up)
        """

        s_val = self.s().detach().to(torch.float32)
        D = float(self.branch)
        s_clamped = torch.minimum(s_val, torch.tensor(D, device=s_val.device, dtype=s_val.dtype))
        s_hard = torch.floor(s_clamped + 0.5).clamp(min=1.0, max=D)
        self._s_hard.copy_(s_hard.to(self._s_hard.dtype))
        self._hard_enabled.fill_(1)
        self._ste_enabled.fill_(0)
        self.s_raw.requires_grad_(False)

        # Ensure inactive branch states are zeroed immediately.
        if self.d_state is not None:
            mask = self.soft_mask(self.d_state.dtype).to(self.d_state.device).unsqueeze(0)
            self.d_state.mul_(mask)

    def alpha_branch(self) -> torch.Tensor:
        return torch.sigmoid(self.tau_n)  # (N,D)

    def beta_soma(self) -> torch.Tensor:
        return torch.sigmoid(self.tau_m)  # (N,)

    # ---------------------------------------------------------------------
    # State + forward
    # ---------------------------------------------------------------------

    def reset_state(self, batch_size: int, device: torch.device, dtype: torch.dtype = torch.float32) -> None:
        self.mem = torch.zeros(batch_size, self.output_dim, device=device, dtype=dtype)
        self.spk = torch.zeros(batch_size, self.output_dim, device=device, dtype=dtype)
        self.d_state = torch.zeros(batch_size, self.output_dim, self.branch, device=device, dtype=dtype)

    def forward_step(self, x_t: torch.Tensor, record: bool = False):
        if self.mem is None or self.spk is None or self.d_state is None:
            self.reset_state(x_t.shape[0], x_t.device, x_t.dtype)

        B = int(x_t.shape[0])

        # soft mask: (1,N,D)
        mask = self.soft_mask(x_t.dtype).to(x_t.device).unsqueeze(0)

        # Branch inputs I_d: (B, N*D) -> (B,N,D)
        I_raw = F.linear(x_t, self.W, self.bias).view(B, self.output_dim, self.branch)
        # For logging/analysis we keep inactive branches at 0 by applying the structural mask once.
        I_eff = I_raw * mask

        alpha = self.alpha_branch().unsqueeze(0)  # (1,N,D)
        # Spec: i_d[t] = M ⊙ ( α ⊙ i_d[t-1] + (1-α) ⊙ I_d[t] )
        # -> apply mask ONCE after the EMA update (do NOT pre-mask I_d, to avoid M^2 on fractional branches).
        self.d_state = alpha * self.d_state + (1.0 - alpha) * I_raw
        self.d_state = self.d_state * mask  # keep inactive branches at 0

        # Soma input H = (1/s) * sum_d i_d
        s_val = self.s().to(x_t.dtype).unsqueeze(0)  # (1,N)
        H = self.d_state.sum(dim=2) / s_val.clamp_min(1e-6)

        beta = self.beta_soma().unsqueeze(0)
        self.mem = self.mem * beta + (1.0 - beta) * H - self.v_th * self.spk
        spk = self.spike_fn(self.mem - self.v_th)
        self.spk = spk

        if not record:
            return spk

        signals = {
            "dendrite_input": I_eff,
            "dendrite_state": self.d_state,
            "soma_input": H,
            "soma_state": self.mem,
            "output": spk,
        }
        return spk, signals

    def forward_sequence(self, x_seq: torch.Tensor, record: bool | Sequence[str] = False):
        # x_seq: (B,T,input_dim)
        B, T, _ = x_seq.shape
        self.reset_state(int(B), x_seq.device, x_seq.dtype)

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
            for t in range(int(T)):
                out_list.append(self.forward_step(x_seq[:, t], record=False))
            return torch.stack(out_list, dim=1)

        # Recording path: stack both outputs and requested signals.
        out_list = []
        rec_lists: Dict[str, list[torch.Tensor]] = {k: [] for k in record_keys}

        for t in range(int(T)):
            y, sig = self.forward_step(x_seq[:, t], record=True)
            out_list.append(y)

            for k in record_keys:
                if k not in sig:
                    raise KeyError(f"Unknown record key: {k!r}")
                rec_lists[k].append(sig[k])

        out_seq = torch.stack(out_list, dim=1)
        rec = {k: torch.stack(rec_lists[k], dim=1) for k in record_keys}
        return out_seq, rec

    def regularization_loss(self, lambda_ortho: float = 0.0, lambda_s: float = 0.0) -> torch.Tensor:
        loss = self.W.new_zeros(())
        if lambda_s != 0.0:
            loss = loss + float(lambda_s) * self.s().mean()
        if lambda_ortho != 0.0:
            alpha = self.alpha_branch()  # (N,D)
            w = self.soft_mask(alpha.dtype).to(alpha.device)  # (N,D)
            loss = loss + float(lambda_ortho) * _ema_orthogonality_loss_weighted(alpha, w)
        return loss

    def get_timing_params(self) -> Dict[str, torch.Tensor]:
        return {
            "alpha": self.alpha_branch().detach().cpu().flatten(),
            "beta": self.beta_soma().detach().cpu().flatten(),
        }

    def get_structure_params(self) -> Dict[str, torch.Tensor]:
        return {
            "s": self.s().detach().cpu(),
            "D_int": self.d_int().detach().cpu(),
        }

    def active_param_count(self) -> int:
        d_int = self.d_int().to(torch.int64)
        active_branches = int(d_int.sum().item())
        active_syn = active_branches * self.input_dim
        if self.bias is not None:
            active_syn += active_branches

        active_tau_n = active_branches
        active_tau_m = self.output_dim
        active_s = self.output_dim
        return int(active_syn + active_tau_n + active_tau_m + active_s)
