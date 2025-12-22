import json
import random
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np
import torch
from matplotlib import pyplot as plt


def set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def accuracy_top1(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()


def _try_reset_state(module: torch.nn.Module, batch_size, device):
    """Attempt to reset state with several common signatures."""

    attempts = []
    if batch_size is not None or device is not None:
        attempts.append(((), {"batch_size": batch_size, "device": device}))
    if batch_size is not None and device is not None:
        attempts.append(((batch_size, device), {}))
    if batch_size is not None:
        attempts.append(((batch_size,), {}))
    if device is not None:
        attempts.append(((), {"device": device}))

    for args, kwargs in attempts:
        try:
            module.reset_state(*args, **kwargs)
            return True
        except TypeError:
            continue
    return False


def reset_states(
    model: torch.nn.Module,
    batch_size: int | None = None,
    device: torch.device | str | None = None,
) -> None:
    for module in model.modules():
        if hasattr(module, "reset"):
            try:
                module.reset()
                continue
            except TypeError:
                last_error = "reset() signature mismatch"
                if hasattr(module, "reset_state") and _try_reset_state(module, batch_size, device):
                    continue
        elif hasattr(module, "reset_state"):
            try:
                module.reset_state()
                continue
            except TypeError:
                if _try_reset_state(module, batch_size, device):
                    continue
                last_error = "reset_state() signature mismatch"
        else:
            continue

        warnings.warn(
            f"State reset failed for module {module.__class__.__name__}: {last_error}",
            RuntimeWarning,
        )
        if hasattr(module, "y"):
            module.y = None


def make_result_dir(exp_name: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("result") / f"{exp_name}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def save_text(path: Path, content: str) -> None:
    path.write_text(content)


def save_json(path: Path, data: Dict[str, Any]) -> None:
    with path.open("w") as f:
        json.dump(data, f, indent=2)


def save_meta(out_dir: Path, start_time: datetime, end_time: datetime, args: Any, summary: Dict[str, Any]) -> None:
    meta_lines = [
        f"start_time: {start_time.isoformat()}",
        f"end_time: {end_time.isoformat()}",
        f"elapsed: {end_time - start_time}",
        f"seed: {args.seed}",
        f"device: {args.device}",
    ]
    for k, v in summary.items():
        meta_lines.append(f"{k}: {v}")
    save_text(out_dir / "meta.txt", "\n".join(meta_lines))


def plot_curves(histories: Dict[str, Iterable[float]], out_png: Path, ylabel: str = "Accuracy") -> None:
    plt.figure()
    for name, values in histories.items():
        plt.plot(list(values), label=name)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def plot_final_bar(acc_dict: Dict[str, float], out_png: Path, ylabel: str = "Accuracy") -> None:
    plt.figure()
    names = list(acc_dict.keys())
    values = [acc_dict[k] for k in names]
    plt.bar(names, values)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()
