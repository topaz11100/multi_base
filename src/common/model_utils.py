from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn


def count_parameters(model: nn.Module) -> int:
    return sum(int(p.numel()) for p in model.parameters())


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(int(p.numel()) for p in model.parameters() if p.requires_grad)


def count_active_parameters(model: nn.Module) -> int:
    """
    Best-effort active parameter counting for variable-branch models.
    If a submodule implements `active_param_count()`, it will be used; otherwise its full params are counted.
    """
    active = 0
    for m in model.modules():
        if hasattr(m, "active_param_count") and callable(getattr(m, "active_param_count")):
            try:
                active += int(m.active_param_count())  # type: ignore
            except Exception:
                pass
    # If nothing contributed, fall back to all parameters
    if active == 0:
        active = count_parameters(model)
    return int(active)


def get_all_weight_tensors(model: nn.Module) -> List[torch.Tensor]:
    weights: List[torch.Tensor] = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim >= 2 and ("weight" in name or name.endswith(".W") or name.endswith(".w")):
            weights.append(p.detach().flatten().cpu())
    return weights


def flatten_tensors(tensors: List[torch.Tensor]) -> torch.Tensor:
    if len(tensors) == 0:
        return torch.zeros(0)
    return torch.cat(tensors, dim=0)


# -----------------------------------------------------------------------------
# Structural parameter utilities (proposed variable-branch models)
# -----------------------------------------------------------------------------


def collect_s_sum_and_count(model: nn.Module) -> Tuple[torch.Tensor, int]:
    """Collect the global sum of all structural parameters `s` in a model.

    Project convention:
      - `s` is per-neuron (shape: (N_neuron,)) for each proposed layer.
      - Global complexity term uses the *global* average over all neurons:

        L_s = (1 / N_total) * sum_over_all_neurons s

    This helper returns (sum_s, N_total). If no `s` tensors exist, returns (0, 0).
    """

    dev = next(model.parameters(), None)
    if dev is None:
        # no parameters: fall back to CPU scalar
        zero = torch.zeros((), dtype=torch.float32)
    else:
        zero = torch.zeros((), device=dev.device, dtype=torch.float32)

    sum_s = zero
    count = 0

    for m in model.modules():
        # Heuristic: proposed layers expose s_raw + s()
        if not (hasattr(m, "s_raw") and hasattr(m, "s") and callable(getattr(m, "s"))):
            continue
        try:
            s = m.s()  # type: ignore
            if not torch.is_tensor(s):
                continue
            if s.numel() == 0:
                continue
            # Only accept floating `s` tensors.
            if not torch.is_floating_point(s):
                continue
            sum_s = sum_s + s.sum()
            count += int(s.numel())
        except Exception:
            continue

    return sum_s, int(count)


def s_complexity_mean(model: nn.Module) -> torch.Tensor:
    """Return the global mean structural complexity term L_s.

    If no proposed layers exist in the model, returns 0.
    """

    sum_s, count = collect_s_sum_and_count(model)
    if count <= 0:
        return sum_s.new_zeros(())
    return sum_s / float(count)


@torch.no_grad()
def harden_variable_branches_(model: nn.Module) -> int:
    """In-place transition for all variable-branch layers in a model.

    A layer is considered variable-branch if it implements a callable `harden_branches()`.
    The transition follows `paper/proposed/varidble_dendric.md`:

      - $s\leftarrow\lfloor\min(D,s)+\tfrac{1}{2}\rfloor$
      - switch mask to hard $\{0,1\}$
      - freeze $s$

    Returns the number of submodules that were hardened.
    """

    hardened = 0
    for m in model.modules():
        if hasattr(m, "harden_branches") and callable(getattr(m, "harden_branches")):
            try:
                m.harden_branches()  # type: ignore[attr-defined]
                hardened += 1
            except Exception:
                continue
    return int(hardened)



@torch.no_grad()
def set_ste_mode_(model: nn.Module, enabled: bool) -> int:
    """Enable/disable STE mode for all variable-branch layers in a model.

    A layer is considered STE-capable if it implements a callable `enable_ste(enabled: bool)`.

    Returns the number of submodules updated.
    """
    count = 0
    for m in model.modules():
        if hasattr(m, "enable_ste") and callable(getattr(m, "enable_ste")):
            try:
                m.enable_ste(bool(enabled))  # type: ignore[attr-defined]
                count += 1
            except Exception:
                continue
    return int(count)



# -----------------------------------------------------------------------------
# Active parameter breakdown (by type/category)
# -----------------------------------------------------------------------------

PARAM_CATEGORIES = (
    "synapse_ff",         # feedforward synapse weights/bias
    "dendrite_to_soma",   # branch->soma mixing (e.g., C, W_mix)
    "timing",             # time constants / decay / coupling / leak factors
    "resonance",          # resonant params (tau, omega)
    "threshold_kernel",   # adaptive threshold kernel params
    "structure",          # structural continuous params (e.g., s)
    "threshold",          # learnable threshold/reset scalars if any
    "other",
)


def layer_active_param_breakdown(layer: nn.Module) -> Dict[str, int]:
    """Best-effort active parameter breakdown for a single neuron layer.

    For variable-branch proposed models, this counts *only active branches*.
    For other layers, it counts all trainable parameters and categorizes them.

    Returns a dict over PARAM_CATEGORIES with integer counts.
    """
    out = {k: 0 for k in PARAM_CATEGORIES}

    # --- Proposed variable-branch models (exact formulas) ---
    d_int_sum: Optional[int] = None
    if hasattr(layer, "d_int") and callable(getattr(layer, "d_int")):
        try:
            d = layer.d_int()  # type: ignore
            if torch.is_tensor(d):
                d_int_sum = int(d.detach().to(torch.int64).sum().item())
            else:
                d_int_sum = int(sum(int(x) for x in d))
        except Exception:
            d_int_sum = None

    # my_DH_SNN: W (+bias), tau_n,tau_m, s
    if d_int_sum is not None and hasattr(layer, "W") and hasattr(layer, "branch"):
        try:
            in_dim = int(getattr(layer, "input_dim"))
            out_dim = int(getattr(layer, "output_dim"))
            # active synapse rows: sum_n d_int[n]
            out["synapse_ff"] += d_int_sum * in_dim
            if getattr(layer, "bias", None) is not None:
                out["synapse_ff"] += d_int_sum
            out["timing"] += d_int_sum        # tau_n (alpha) per active branch
            out["timing"] += out_dim          # tau_m (beta) per soma
            out["structure"] += out_dim       # s (per neuron)
            return {k: int(v) for k, v in out.items()}
        except Exception:
            pass

    # my_R_DH_SNN: soma-dense W_in, W_mix, tau_n,tau_m, s
    if d_int_sum is not None and hasattr(layer, "W_mix") and hasattr(layer, "branch"):
        try:
            out_dim = int(getattr(layer, "output_dim"))
            in_dim = int(getattr(layer, "input_dim"))
            # soma-dense layer connection weights W_in (always active)
            if hasattr(layer, "W_in"):
                out["synapse_ff"] += out_dim * in_dim
            elif hasattr(layer, "w_in"):
                # legacy (axon-shared) fallback
                out["synapse_ff"] += in_dim
            out["dendrite_to_soma"] += d_int_sum
            out["timing"] += d_int_sum
            out["timing"] += out_dim
            out["structure"] += out_dim
            return {k: int(v) for k, v in out.items()}
        except Exception:
            pass

    # my_D_RF: fc params (always active), tau/omega for active branches, a_k kernel, s
    if d_int_sum is not None and hasattr(layer, "tau_raw") and hasattr(layer, "omega_raw") and hasattr(layer, "a_raw") and hasattr(layer, "fc"):
        try:
            out_dim = int(getattr(layer, "output_dim"))
            # synapse always active
            fc = getattr(layer, "fc")
            if isinstance(fc, nn.Linear):
                out["synapse_ff"] += int(sum(int(p.numel()) for p in fc.parameters() if p.requires_grad))
            else:
                out["synapse_ff"] += int(sum(int(p.numel()) for p in layer.parameters() if p.requires_grad))
            out["resonance"] += 2 * d_int_sum  # tau + omega (per active branch)
            th_len = int(getattr(layer, "th_len", int(getattr(layer, "a_raw").numel())))
            out["threshold_kernel"] += th_len
            out["structure"] += out_dim
            return {k: int(v) for k, v in out.items()}
        except Exception:
            pass

    # --- Baseline D-RF (fixed branches) ---
    if hasattr(layer, "tau_raw") and hasattr(layer, "omega_raw") and hasattr(layer, "alpha_th_raw") and hasattr(layer, "C") and hasattr(layer, "fc"):
        try:
            out_dim = int(getattr(layer, "output_dim"))
            D = int(getattr(layer, "branch"))
            fc = getattr(layer, "fc")
            if isinstance(fc, nn.Linear):
                out["synapse_ff"] += int(sum(int(p.numel()) for p in fc.parameters() if p.requires_grad))
            out["resonance"] += 2 * out_dim * D
            out["dendrite_to_soma"] += out_dim * D  # C weights
            th_len = int(getattr(layer, "th_len", int(getattr(layer, "alpha_th_raw").numel())))
            out["threshold_kernel"] += th_len
            return {k: int(v) for k, v in out.items()}
        except Exception:
            pass

    # --- Baseline DH-SNN (fixed branches) ---
    if hasattr(layer, "tau_n") and hasattr(layer, "tau_m") and hasattr(layer, "branch") and hasattr(layer, "fc"):
        try:
            out_dim = int(getattr(layer, "output_dim"))
            D = int(getattr(layer, "branch"))
            fc = getattr(layer, "fc")
            if isinstance(fc, nn.Linear):
                out["synapse_ff"] += int(sum(int(p.numel()) for p in fc.parameters() if p.requires_grad))
            out["timing"] += out_dim * D  # tau_n
            out["timing"] += out_dim      # tau_m
            return {k: int(v) for k, v in out.items()}
        except Exception:
            pass

    # --- Heuristic categorization for all other layers ---
    for name, p in layer.named_parameters(recurse=False):
        if not p.requires_grad:
            continue
        n = name.lower()
        numel = int(p.numel())

        if "weight" in n or n in ("w", "w_ff", "w_in") or n.endswith(".w"):
            out["synapse_ff"] += numel
        elif "bias" in n:
            out["synapse_ff"] += numel
        elif "w_mix" in n or n == "c" or n.endswith(".c") or n == "c_raw":
            out["dendrite_to_soma"] += numel
        elif "omega" in n:
            out["resonance"] += numel
        elif "tau" in n:
            # For non D-RF layers, tau is usually a timing/leak factor.
            out["timing"] += numel
        elif "alpha_th" in n or (n.startswith("a") and "raw" in n) or "th_kernel" in n:
            out["threshold_kernel"] += numel
        elif "alpha" in n or "beta" in n or "decay" in n or "gamma" in n or "kappa" in n:
            out["timing"] += numel
        elif "s_raw" in n or n == "s":
            out["structure"] += numel
        elif "v_th" in n or "vreset" in n or "v_reset" in n:
            out["threshold"] += numel
        else:
            out["other"] += numel

    # If everything fell into "other" only, collapse into "other" as-is.
    return {k: int(v) for k, v in out.items()}


def breakdown_total(breakdown: Dict[str, int]) -> int:
    return int(sum(int(v) for v in breakdown.values()))


def aggregate_breakdowns(breakdowns: List[Dict[str, int]]) -> Dict[str, int]:
    out = {k: 0 for k in PARAM_CATEGORIES}
    for b in breakdowns:
        for k in out:
            out[k] += int(b.get(k, 0))
    return {k: int(v) for k, v in out.items()}


def format_breakdown_table(b: Dict[str, int], prefix: str = "") -> str:
    lines = []
    keys = [k for k in PARAM_CATEGORIES if b.get(k, 0) != 0]
    if not keys:
        keys = list(PARAM_CATEGORIES)
    width = max(len(k) for k in keys) if keys else 10
    for k in keys:
        lines.append(f"{prefix}{k:<{width}} : {int(b.get(k,0))}")
    lines.append(f"{prefix}{'total':<{width}} : {breakdown_total(b)}")
    return "\n".join(lines)
