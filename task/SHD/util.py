import json
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Mapping

import matplotlib.pyplot as plt
import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_results_dir(root: str, exp_name: str, timestamp: datetime | None = None) -> Path:
    timestamp = timestamp or datetime.now()
    out_dir = Path(root) / f"{exp_name}_{timestamp.strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def dump_text(path: Path, filename: str, text: str | Mapping | Iterable) -> None:
    path.mkdir(parents=True, exist_ok=True)
    file_path = path / filename
    if isinstance(text, Mapping):
        content = json.dumps(text, indent=2)
    elif isinstance(text, Iterable) and not isinstance(text, (str, bytes)):
        content = "\n".join(str(t) for t in text)
    else:
        content = str(text)
    file_path.write_text(content)


def _add_dh_params(module: torch.nn.Module, counted: set) -> int:
    total = 0
    dense = getattr(module, "dense", None)
    mask = getattr(module, "mask", None)
    if dense is None:
        return 0
    if dense.weight is not None and dense.weight.requires_grad:
        counted.add(dense.weight)
        if mask is not None:
            total += int(mask.sum().item())
        else:
            total += dense.weight.numel()
    if dense.bias is not None and dense.bias.requires_grad:
        counted.add(dense.bias)
        total += dense.bias.numel()
    for name, param in module.named_parameters(recurse=False):
        if name == "dense":
            continue
        if param.requires_grad and param not in counted:
            total += param.numel()
            counted.add(param)
    return total


def count_effective_params(model: torch.nn.Module) -> int:
    counted: set[torch.nn.Parameter] = set()
    total = 0
    for module in model.modules():
        if hasattr(module, "mask") and hasattr(module, "dense"):
            total += _add_dh_params(module, counted)
    for param in model.parameters():
        if param.requires_grad and param not in counted:
            total += param.numel()
            counted.add(param)
    return total


def plot_accuracy_curves(histories: Dict[str, Dict[str, list[float]]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    colors = plt.cm.tab10(np.linspace(0, 1, len(histories)))
    for (model, history), color in zip(histories.items(), colors):
        epochs = np.arange(1, len(history.get("train_acc", [])) + 1)
        plt.figure()
        plt.plot(epochs, history.get("train_acc", []), label="train", color=color)
        plt.plot(epochs, history.get("test_acc", []), label="test", linestyle="--", color=color)
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title(f"Accuracy - {model}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"acc_curve_{model}.png")
        plt.close()

    plt.figure()
    for (model, history), color in zip(histories.items(), colors):
        epochs = np.arange(1, len(history.get("train_acc", [])) + 1)
        plt.plot(epochs, history.get("train_acc", []), label=f"{model} train", color=color)
        plt.plot(epochs, history.get("test_acc", []), label=f"{model} test", linestyle="--", color=color)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "acc_curve_all.png")
    plt.close()

    plt.figure()
    labels = []
    finals = []
    for model, history in histories.items():
        labels.append(model)
        finals.append(history.get("best_test", 0))
    plt.bar(labels, finals)
    plt.ylabel("Best Test Accuracy")
    plt.title("Best Test Accuracy per Model")
    plt.tight_layout()
    plt.savefig(out_dir / "acc_final_bar.png")
    plt.close()
