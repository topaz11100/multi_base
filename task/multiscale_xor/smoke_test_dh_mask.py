"""Quick smoke test ensuring DH masks remain enforced during training."""

from __future__ import annotations

import torch

from model import build_model
from run import enforce_structural_masks


def _assert_mask_enforced(model: torch.nn.Module, eps: float = 0.0) -> None:
    dense = model.dense_1.dense
    mask = model.dense_1.mask
    max_abs = (dense.weight * (1 - mask)).abs().max().item()
    assert max_abs <= max(eps, 0.0), f"Mask violation detected: {max_abs}"


def main() -> None:
    device = torch.device("cpu")
    model = build_model(
        "dh",
        input_dim=4,
        hidden_dim=8,
        output_dim=2,
        device=device,
        neuron_kwargs={"branch": 1},
    )

    # Initial mask application enforced at build time
    _assert_mask_enforced(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    x = torch.rand(2, 3, 4, device=device)
    y = torch.tensor([0, 1], device=device)

    logits = model(x)
    loss = torch.nn.functional.cross_entropy(logits.mean(dim=1), y)
    loss.backward()
    optimizer.step()

    enforce_structural_masks(model, check=True)
    _assert_mask_enforced(model)
    print("Mask enforcement smoke test passed.")


if __name__ == "__main__":
    main()
