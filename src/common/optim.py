from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn


@dataclass(frozen=True)
class AdamWGroupInfo:
    """Summary of AdamW parameter grouping for weight decay."""

    weight_decay: float
    weight_decay_dend_soma: Optional[float]
    num_decay_layer_params: int
    num_decay_dend_soma_params: int
    num_no_decay_params: int


def _collect_decay_param_ids(model: nn.Module) -> Tuple[Set[int], Set[int]]:
    """Return (layer_weight_ids, dend_soma_weight_ids).

    Policy:
    - Layer connection weights (weight decay 대상):
      - nn.Linear.weight
      - custom parameters named W (my_DH_SNN) and W_in (my_R_DH_SNN)
    - Dendrite->soma mixing weights (my_R_DH_SNN only):
      - custom parameter named W_mix

    All other parameters (biases, timing factors, s, etc.) are excluded from weight decay.
    """

    layer_ids: Set[int] = set()
    ds_ids: Set[int] = set()

    # 1) Standard dense connections
    for m in model.modules():
        if isinstance(m, nn.Linear):
            if getattr(m, "weight", None) is not None:
                layer_ids.add(id(m.weight))

    # 2) Custom layer parameters (project-specific)
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.endswith(".W") or name == "W":
            layer_ids.add(id(p))
        if name.endswith(".W_in") or name == "W_in":
            layer_ids.add(id(p))
        if name.endswith(".W_mix") or name == "W_mix":
            ds_ids.add(id(p))

    # Ensure W_mix is never treated as a layer connection weight.
    layer_ids.difference_update(ds_ids)
    return layer_ids, ds_ids


def make_adamw_param_groups(
    model: nn.Module,
    weight_decay: float,
    weight_decay_dend_soma: Optional[float] = None,
) -> Tuple[List[Dict[str, Any]], AdamWGroupInfo]:
    """Create AdamW parameter groups matching the project weight-decay spec.

    Args:
        model: Target model.
        weight_decay: Weight decay coefficient for **layer connection weights**.
        weight_decay_dend_soma: Optional separate coefficient for my_R_DH_SNN dendrite->soma
            mixing weights (W_mix). If None, it follows `weight_decay`.

    Returns:
        (param_groups, info)
    """

    wd = float(weight_decay)
    wd_ds = wd if weight_decay_dend_soma is None else float(weight_decay_dend_soma)

    layer_ids, ds_ids = _collect_decay_param_ids(model)

    decay_layer: List[torch.nn.Parameter] = []
    decay_ds: List[torch.nn.Parameter] = []
    no_decay: List[torch.nn.Parameter] = []

    for _, p in model.named_parameters():
        if not p.requires_grad:
            continue
        pid = id(p)
        if pid in ds_ids:
            decay_ds.append(p)
        elif pid in layer_ids:
            decay_layer.append(p)
        else:
            no_decay.append(p)

    param_groups: List[Dict[str, Any]] = []
    if len(decay_layer) > 0:
        param_groups.append({"params": decay_layer, "weight_decay": wd})
    if len(decay_ds) > 0:
        param_groups.append({"params": decay_ds, "weight_decay": wd_ds})
    if len(no_decay) > 0:
        param_groups.append({"params": no_decay, "weight_decay": 0.0})

    info = AdamWGroupInfo(
        weight_decay=wd,
        weight_decay_dend_soma=(wd_ds if len(decay_ds) > 0 else None),
        num_decay_layer_params=len(decay_layer),
        num_decay_dend_soma_params=len(decay_ds),
        num_no_decay_params=len(no_decay),
    )
    return param_groups, info


def build_adamw(
    model: nn.Module,
    lr: float,
    weight_decay: float = 0.0,
    weight_decay_dend_soma: Optional[float] = None,
) -> Tuple[torch.optim.Optimizer, AdamWGroupInfo]:
    """Build AdamW optimizer with correct weight-decay grouping."""

    groups, info = make_adamw_param_groups(
        model,
        weight_decay=float(weight_decay),
        weight_decay_dend_soma=weight_decay_dend_soma,
    )
    # IMPORTANT: keep global weight_decay=0 to prevent accidental decay on the no_decay group.
    opt = torch.optim.AdamW(groups, lr=float(lr), weight_decay=0.0)
    return opt, info
