from __future__ import annotations

from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.common.surrogate import SpikeFn


class DRFDenseLayer(nn.Module):
    """
    Baseline Dendritic Resonate-and-Fire (D-RF) dense layer, implemented from the D-RF paper.

    - Synapse: I[t] = W x[t]
    - Dendritic branches (complex state):
        Z_d[t] = exp(Δ D_d) Z_d[t-1] + Γ_d I[t]
        D_d = -1/τ_d + i ω_d
        Γ_d = (exp(Δ D_d) - 1) / D_d   (ZOH discretization)

    - Soma drive:
        H[t] = sum_d C_d * Re{Z_d[t]}

    - Adaptive threshold (paper Eq. 11/12; simplified implementation):
        p[t] = Θ(H[t] - V_pre)            (pre-activation indicator)
        V_th[t] = V_pre + sum_{k=1..K} α_k p[t-k]
        S[t] = Θ(H[t] - V_th[t])
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        branch: int = 8,
        th_len: int = 4,
        delta: float = 1.0,
        v_pre: float = 1.0,
        bias: bool = True,
        spike_fn: Optional[SpikeFn] = None,
        tau_min: float = 1e-3,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.branch = int(branch)
        self.th_len = int(th_len)
        self.delta = float(delta)
        self.v_pre = float(v_pre)
        self.tau_min = float(tau_min)

        self.spike_fn = spike_fn or SpikeFn(name="mg", lens=0.5, gamma=0.5)

        self.fc = nn.Linear(self.input_dim, self.output_dim, bias=bias)

        # Branch parameters
        self.tau_raw = nn.Parameter(torch.full((self.output_dim, self.branch), 2.0))
        self.omega_raw = nn.Parameter(torch.full((self.output_dim, self.branch), 1.0))

        # Soma mixing weights C (learnable)
        self.C = nn.Parameter(torch.full((self.output_dim, self.branch), 1.0 / float(self.branch)))

        # Adaptive threshold kernel α_k in (0,1)
        self.alpha_th_raw = nn.Parameter(torch.zeros(self.th_len))

        # States: complex Z = u + i v
        self.u: Optional[torch.Tensor] = None  # (B,N,D)
        self.v: Optional[torch.Tensor] = None  # (B,N,D)
        self.pre_hist: Optional[torch.Tensor] = None  # (B,N,K)

    def tau(self) -> torch.Tensor:
        return F.softplus(self.tau_raw) + self.tau_min

    def omega(self) -> torch.Tensor:
        return F.softplus(self.omega_raw)

    def alpha_th(self) -> torch.Tensor:
        return torch.sigmoid(self.alpha_th_raw)

    def reset_state(self, batch_size: int, device: torch.device, dtype: torch.dtype = torch.float32) -> None:
        self.u = torch.zeros(batch_size, self.output_dim, self.branch, device=device, dtype=dtype)
        self.v = torch.zeros(batch_size, self.output_dim, self.branch, device=device, dtype=dtype)
        self.pre_hist = torch.zeros(batch_size, self.output_dim, self.th_len, device=device, dtype=dtype)

    def _compute_rho_gamma(self, tau: torch.Tensor, omega: torch.Tensor):
        # tau, omega: (N,D)
        delta = self.delta
        r = torch.exp(-delta / tau)  # (N,D)
        theta = omega * delta
        rho_real = r * torch.cos(theta)
        rho_imag = r * torch.sin(theta)

        # D = a + i b
        a = -1.0 / tau
        b = omega

        r1 = rho_real - 1.0
        r2 = rho_imag

        denom = a * a + b * b + 1e-12

        gamma_real = (r1 * a + r2 * b) / denom
        gamma_imag = (r2 * a - r1 * b) / denom
        return rho_real, rho_imag, gamma_real, gamma_imag

    def forward_step(self, x_t: torch.Tensor, record: bool = False):
        if self.u is None or self.v is None or self.pre_hist is None:
            self.reset_state(x_t.shape[0], x_t.device, x_t.dtype)

        I_t = self.fc(x_t)  # (B,N)

        tau = self.tau()  # (N,D)
        omega = self.omega()  # (N,D)
        rho_r, rho_i, gam_r, gam_i = self._compute_rho_gamma(tau, omega)

        # Expand for batch
        rho_r_b = rho_r.unsqueeze(0)
        rho_i_b = rho_i.unsqueeze(0)
        gam_r_b = gam_r.unsqueeze(0)
        gam_i_b = gam_i.unsqueeze(0)

        I_b = I_t.unsqueeze(-1)  # (B,N,1)

        u_new = rho_r_b * self.u - rho_i_b * self.v + gam_r_b * I_b
        v_new = rho_i_b * self.u + rho_r_b * self.v + gam_i_b * I_b

        self.u = u_new
        self.v = v_new

        H_t = (self.C.unsqueeze(0) * self.u).sum(dim=2)  # (B,N)

        # Pre-indicator for threshold update (uses V_pre)
        pre = self.spike_fn(H_t - self.v_pre)  # (B,N)

        # Threshold uses previous pre values
        a_th = self.alpha_th()  # (K,)
        V_th = self.v_pre + (self.pre_hist * a_th.view(1, 1, -1)).sum(dim=2)  # (B,N)

        spk = self.spike_fn(H_t - V_th)

        # Update history: shift right and insert current pre at index 0
        self.pre_hist = torch.cat([pre.unsqueeze(-1), self.pre_hist[:, :, :-1]], dim=2)

        if not record:
            return spk

        signals = {
            "dendrite_input": I_t.unsqueeze(-1).expand(-1, -1, self.branch),
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
    def get_timing_params(self) -> Dict[str, torch.Tensor]:
        return {
            "tau": self.tau().detach().cpu().flatten(),
            "omega": self.omega().detach().cpu().flatten(),
        }

    def active_param_count(self) -> int:
        return sum(int(p.numel()) for p in self.parameters())
