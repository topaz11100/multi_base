from __future__ import annotations

"""Readout functions for SNN classifiers.

This codebase previously supported multiple readout / encoding schemes (e.g., rate and
first-spike latency). Those options were removed because they frequently create an
objective mismatch with `nn.CrossEntropyLoss`.

The project now uses **membrane-potential readout only**:

    logits = mean_t( soma_state[t] )

where `soma_state` is the output-layer membrane / soma state recorded during the
forward pass.

Implementation note:
- For neuron models that apply a reset at the spike time, `soma_state` should be
  recorded **before** the reset is applied (pre-reset membrane). Otherwise, the
  readout can lose decision-relevant information.
"""

from typing import Optional

import torch


def membrane_readout(soma_seq: torch.Tensor) -> torch.Tensor:
    """Membrane readout.

    Args:
        soma_seq: Tensor of shape (B, T, C) from the output layer's `soma_state`.

    Returns:
        logits: Tensor of shape (B, C).
    """
    if soma_seq is None:
        raise ValueError("membrane_readout requires soma_seq (got None)")
    if soma_seq.dim() != 3:
        raise ValueError(f"soma_seq must be (B,T,C), got {tuple(soma_seq.shape)}")
    return soma_seq.to(torch.float32).mean(dim=1)


def apply_readout(*, soma_seq: Optional[torch.Tensor]) -> torch.Tensor:
    """Apply membrane readout.

    Kept as a thin wrapper so call sites are explicit about passing soma_state.
    """
    if soma_seq is None:
        raise ValueError("apply_readout requires soma_seq (output layer soma_state)")
    return membrane_readout(soma_seq)
