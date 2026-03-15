from __future__ import annotations

from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.common.surrogate import SpikeFn


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
    """
    if s.ndim != 1:
        raise ValueError(f"s must be 1D (got shape={tuple(s.shape)})")
    D = int(branch)
    s_c = s.clamp_min(0.0).clamp_max(float(D))
    D_hard = torch.floor(s_c + 0.5).to(torch.int64).clamp(min=1, max=D)
    idx = torch.arange(D, device=s.device, dtype=torch.int64).view(1, D)
    mask_hard = (idx < D_hard.view(-1, 1)).to(s.dtype)
    return mask_hard, D_hard


def _drf_orthogonality_loss_weighted(
    tau: torch.Tensor, omega: torch.Tensor, w: torch.Tensor, delta: float = 1.0
) -> torch.Tensor:
    """Weighted D-RF orthogonality loss (mean over neurons).

    Applies per-branch weights (soft mask) w_i w_j before summing i<j.
    """
    if tau.ndim != 2 or omega.ndim != 2 or w.ndim != 2 or tau.shape != omega.shape or tau.shape != w.shape:
        raise ValueError("tau, omega, w must all be (N,D) with the same shape")
    _, D = tau.shape
    if int(D) <= 1:
        return tau.new_zeros(())
    t = tau.clamp_min(1e-6)
    r = torch.exp(-float(delta) / t)
    ri = r.unsqueeze(2)
    rj = r.unsqueeze(1)
    wi = omega.unsqueeze(2)
    wj = omega.unsqueeze(1)
    m = ri * rj
    denom = 1.0 + m * m - 2.0 * m * torch.cos(float(delta) * (wi - wj))
    denom = denom.clamp_min(1e-6)
    num = (1.0 - ri * ri) * (1.0 - rj * rj)
    cos2 = (num / denom) * (w.unsqueeze(2) * w.unsqueeze(1))
    return torch.triu(cos2, diagonal=1).sum(dim=(1, 2)).mean()


class MyDRFDenseLayer(nn.Module):
    """Proposed D-RF neuron (paper/proposed/my_D_RF_neuron.md).

    Differences vs baseline:
      - Soma input H[t] is the average of active dendrites (no C weights).
      - Adaptive threshold uses past output spikes:
            V_th[t] = V_pre + sum_k a_k p[t-k]
      - Variable dendrite count s with active D_int=floor(s).

    Updated requirement:
      - branch tensor size is fixed by `branch`
      - continuous s is constrained only by (S_min, S_max)
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        branch: int = 8,
        S_min: float = 1.0,
        S_max: Optional[float] = None,
        th_len: int = 4,
        delta: float = 1.0,
        v_pre: float = 1.0,
        bias: bool = True,
        spike_fn: Optional[SpikeFn] = None,
        tau_min: float = 1e-3,
        s_init: Optional[float] = None,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.branch = int(branch)
        self.th_len = int(th_len)
        self.delta = float(delta)
        self.v_pre = float(v_pre)
        self.tau_min = float(tau_min)

        if S_max is None:
            S_max = float(self.branch)

        self.S_min = float(S_min)
        self.S_max = float(S_max)

        if not (1 <= self.branch):
            raise ValueError(f"branch must be >= 1 (got {self.branch})")
        if not (1.0 <= self.S_min <= self.S_max):
            raise ValueError(f"Require 1 <= S_min <= S_max (got S_min={self.S_min}, S_max={self.S_max})")

        self.spike_fn = spike_fn or SpikeFn(name="mg", lens=0.5, gamma=0.5)

        # Synapse: I[t] = W x[t]
        self.fc = nn.Linear(self.input_dim, self.output_dim, bias=bias)

        # Branch parameters
        self.tau_raw = nn.Parameter(torch.full((self.output_dim, self.branch), 2.0))
        self.omega_raw = nn.Parameter(torch.full((self.output_dim, self.branch), 1.0))

        # Adaptive threshold kernel a_k > 0
        self.a_raw = nn.Parameter(torch.zeros(self.th_len))

        # Structure parameter s in [S_min,S_max] (per-neuron).
# Default policy: start from the *S_min side* (sparse branches) and let training grow s if needed.
# NOTE: We still clamp the normalized ratio to [eps, 1-eps] to avoid sigmoid saturation.
        # Default policy: start from the *S_min side* (sparse branches) and let training grow s if needed.
        # NOTE: We still clamp the normalized ratio to [eps, 1-eps] to avoid sigmoid saturation.
        if s_init is None:
            s_init = float(self.S_min)
        s0 = torch.full((self.output_dim,), float(s_init), dtype=torch.float32)
        if abs(self.S_max - self.S_min) < 1e-12:
            s_norm = torch.zeros_like(s0)
        else:
            s_norm = (s0 - float(self.S_min)) / float(self.S_max - self.S_min)
        # Avoid sigmoid saturation at exactly 0/1.
        eps = 1e-2
        s_norm = s_norm.clamp(eps, 1.0 - eps)
        s_raw_init = torch.log(s_norm / (1.0 - s_norm))
        self.s_raw = nn.Parameter(s_raw_init.clone().detach())

        # Two-stage training support (soft mask -> hard mask).
        # When hardened, we freeze s and use an integer-valued s_hard = floor(min(D, s) + 1/2).
        self.register_buffer("_hard_enabled", torch.zeros((), dtype=torch.uint8), persistent=True)
        self.register_buffer(
            "_s_hard",
            torch.full((self.output_dim,), float(self.S_min), dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer("_ste_enabled", torch.zeros((), dtype=torch.uint8), persistent=False)

        # States
        self.u: Optional[torch.Tensor] = None  # (B,N,D)
        self.v: Optional[torch.Tensor] = None  # (B,N,D)
        # Pre-indicator history (original D-RF style).
        # p[t] = Θ(H[t] - V_pre)
        # V_th[t] = V_pre + sum_k a_k p[t-k]
        self.p_hist: Optional[torch.Tensor] = None  # (B,N,K)

    # ---------------------------------------------------------------------
    # Structure helpers
    # ---------------------------------------------------------------------

    def s(self) -> torch.Tensor:
        # After hardening, s becomes an integer (float tensor) and is frozen.
        if int(self._hard_enabled.item()) == 1:
            return self._s_hard.to(device=self.s_raw.device, dtype=self.s_raw.dtype)
        if abs(self.S_max - self.S_min) < 1e-12:
            return torch.full((self.output_dim,), float(self.S_max), device=self.s_raw.device, dtype=self.s_raw.dtype)
        return self.S_min + (self.S_max - self.S_min) * torch.sigmoid(self.s_raw)

    def d_int(self) -> torch.Tensor:
        """Per-neuron integer number of active branches in the *current* masking mode."""
        if int(self._hard_enabled.item()) == 1:
            return self._s_hard.to(device=self.s_raw.device, dtype=torch.int64).clamp(min=1, max=int(self.branch))
        if int(self._ste_enabled.item()) == 1:
            _, D_hard = _hard_branch_mask_from_s(self.s().to(torch.float32), self.branch)
            return D_hard
        _, D_int = _soft_branch_mask_from_s(self.s(), self.branch)
        return D_int

    def soft_mask(self, dtype: torch.dtype) -> torch.Tensor:
        """Per-neuron mask M(s) with shape (N,D)."""
        if int(self._hard_enabled.item()) == 1:
            D_int = self.d_int().to(torch.int64)
            idx = torch.arange(int(self.branch), device=D_int.device, dtype=torch.int64).view(1, int(self.branch))
            return (idx < D_int.view(-1, 1)).to(dtype)

        s = self.s().to(dtype)
        mask_soft, _ = _soft_branch_mask_from_s(s, self.branch)

        if int(self._ste_enabled.item()) == 1:
            mask_hard, _ = _hard_branch_mask_from_s(s, self.branch)
            # Straight-Through Estimator (STE): forward=hard, backward=soft
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
        """Transition: soft mask -> hard mask.

        Implements the doc rule:
          s <- floor(min(D, s) + 1/2) and freeze s.
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
        if self.u is not None:
            m = self.soft_mask(self.u.dtype).to(self.u.device).unsqueeze(0)
            self.u.mul_(m)
        if self.v is not None:
            m = self.soft_mask(self.v.dtype).to(self.v.device).unsqueeze(0)
            self.v.mul_(m)

    # ---------------------------------------------------------------------
    # Parameter transforms
    # ---------------------------------------------------------------------

    def tau(self) -> torch.Tensor:
        return F.softplus(self.tau_raw) + self.tau_min

    def omega(self) -> torch.Tensor:
        return F.softplus(self.omega_raw)

    def a_kernel(self) -> torch.Tensor:
        return F.softplus(self.a_raw)

    # ---------------------------------------------------------------------
    # State + forward
    # ---------------------------------------------------------------------

    def reset_state(self, batch_size: int, device: torch.device, dtype: torch.dtype = torch.float32) -> None:
        self.u = torch.zeros(batch_size, self.output_dim, self.branch, device=device, dtype=dtype)
        self.v = torch.zeros(batch_size, self.output_dim, self.branch, device=device, dtype=dtype)
        self.p_hist = torch.zeros(batch_size, self.output_dim, self.th_len, device=device, dtype=dtype)

    def _compute_rho_gamma(self, tau: torch.Tensor, omega: torch.Tensor):
        delta = self.delta
        r = torch.exp(-delta / tau)
        theta = omega * delta
        rho_real = r * torch.cos(theta)
        rho_imag = r * torch.sin(theta)

        a = -1.0 / tau
        b = omega
        r1 = rho_real - 1.0
        r2 = rho_imag
        denom = a * a + b * b + 1e-12
        gamma_real = (r1 * a + r2 * b) / denom
        gamma_imag = (r2 * a - r1 * b) / denom
        return rho_real, rho_imag, gamma_real, gamma_imag

    def forward_step(self, x_t: torch.Tensor, record: bool = False):
        if self.u is None or self.v is None or self.p_hist is None:
            self.reset_state(x_t.shape[0], x_t.device, x_t.dtype)

        # soft mask: (1,N,D)
        mask = self.soft_mask(x_t.dtype).to(x_t.device).unsqueeze(0)

        I_t = self.fc(x_t)  # (B,N)

        tau = self.tau()
        omega = self.omega()
        rho_r, rho_i, gam_r, gam_i = self._compute_rho_gamma(tau, omega)

        I_b = I_t.unsqueeze(-1)

        u_new = rho_r.unsqueeze(0) * self.u - rho_i.unsqueeze(0) * self.v + gam_r.unsqueeze(0) * I_b
        v_new = rho_i.unsqueeze(0) * self.u + rho_r.unsqueeze(0) * self.v + gam_i.unsqueeze(0) * I_b

        self.u = u_new * mask
        self.v = v_new * mask

        s_val = self.s().to(x_t.dtype).unsqueeze(0)  # (1,N)
        H_t = self.u.sum(dim=2) / s_val.clamp_min(1e-6)  # (B,N)

        # Pre-indicator (original D-RF style)
        # p[t] = Θ(H[t] - V_pre)
        p = self.spike_fn(H_t - float(self.v_pre))

        # Adaptive threshold from past pre-indicators
        a = self.a_kernel().to(x_t.dtype)  # (K,)
        V_th = self.v_pre + (self.p_hist * a.view(1, 1, -1)).sum(dim=2)

        spk = self.spike_fn(H_t - V_th)

        # update history with current pre-indicator
        self.p_hist = torch.cat([p.unsqueeze(-1), self.p_hist[:, :, :-1]], dim=2)

        if not record:
            return spk

        signals = {
            "dendrite_input": I_t.unsqueeze(-1).expand(-1, -1, self.branch) * mask,
            "dendrite_state": self.u,
            "soma_input": H_t,
            "soma_state": H_t,
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

    # ---------------------------------------------------------------------
    # Regularization / logging helpers
    # ---------------------------------------------------------------------

    def regularization_loss(self, lambda_ortho: float = 0.0, lambda_s: float = 0.0) -> torch.Tensor:
        loss = self.fc.weight.new_zeros(())
        if lambda_s != 0.0:
            loss = loss + float(lambda_s) * self.s().mean()
        if lambda_ortho != 0.0:
            tau = self.tau()
            omega = self.omega()
            w = self.soft_mask(tau.dtype).to(tau.device)
            loss = loss + float(lambda_ortho) * _drf_orthogonality_loss_weighted(tau, omega, w, delta=self.delta)
        return loss

    def get_timing_params(self) -> Dict[str, torch.Tensor]:
        return {
            "tau": self.tau().detach().cpu().flatten(),
            "omega": self.omega().detach().cpu().flatten(),
        }

    def get_structure_params(self) -> Dict[str, torch.Tensor]:
        return {
            "s": self.s().detach().cpu(),
            "D_int": self.d_int().detach().cpu(),
        }

    def active_param_count(self) -> int:
        d_int = self.d_int().to(torch.int64)
        active_branches = int(d_int.sum().item())
        # Synapse always active
        active_syn = sum(int(p.numel()) for p in self.fc.parameters())
        # branch params active only for first d_int branches
        active_tau = active_branches
        active_omega = active_branches
        active_a = self.th_len
        active_s = self.output_dim
        return int(active_syn + active_tau + active_omega + active_a + active_s)
