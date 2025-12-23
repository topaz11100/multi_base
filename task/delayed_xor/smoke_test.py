import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT))
sys.path.append(str(Path(__file__).resolve().parent))

from data import generate_batch
from model import get_model
from run import compute_loss_and_acc
from util import apply_masks, collect_mask_modules


@torch.no_grad()
def _assert_output_shapes(logits: torch.Tensor, batch: torch.Tensor) -> None:
    assert logits.shape[:2] == batch.shape[:2], "Time and batch dims should match"
    assert logits.device == batch.device, "Output device mismatch"
    assert logits.dtype.is_floating_point, "Logits should be floating point"


def run_once(model_name: str, device: torch.device) -> None:
    model = get_model(
        model_name,
        input_dim=20,
        hidden_dim=8,
        output_dim=2,
        device=device,
        dh_branches=1,
        dh_readout="linear",
    )
    mask_modules = collect_mask_modules(model)
    apply_masks(mask_modules)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for step in range(2):
        batch, targets, mask = generate_batch(
            batch_size=4,
            delay=4,
            time_steps=30,
            channel_size=20,
            coding_time=5,
            device=device,
            target_mode="last_step",
        )
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch)
        _assert_output_shapes(logits, batch)
        loss, acc = compute_loss_and_acc(logits, targets, mask)
        assert torch.isfinite(loss), "Loss must be finite"
        loss.backward()
        optimizer.step()
        apply_masks(mask_modules)
        assert acc >= 0.0, "Accuracy should be non-negative"


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test for delayed XOR models")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but not available")

    for name in ("cp", "tc", "ts", "dh"):
        run_once(name, device)
    print(f"Smoke test passed on device: {device}")


if __name__ == "__main__":
    main()
