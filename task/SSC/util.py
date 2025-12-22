import json
import random
from datetime import datetime
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def count_trainable_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def accuracy_top1(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()


def make_result_dir(exp_name: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("result") / f"{exp_name}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def save_meta(path: Path, args, extra: Dict) -> None:
    meta_path = path / "meta.txt"
    lines = []
    lines.append(f"start_time: {extra.get('start_time', '')}")
    lines.append(f"end_time: {extra.get('end_time', '')}")
    lines.append(f"elapsed: {extra.get('elapsed', '')}")
    lines.append(f"seed: {getattr(args, 'seed', '')}")
    lines.append(f"device: {getattr(args, 'device', '')}")
    lines.append(f"command: {extra.get('command', '')}")
    lines.append("args:")
    for k, v in sorted(vars(args).items()):
        lines.append(f"  {k}: {v}")
    meta_path.write_text("\n".join(lines))
    json_path = path / "args.json"
    with json_path.open("w") as f:
        json.dump(vars(args), f, indent=2)


def save_accuracy_txt(path: Path, neuron_name: str, metrics: Dict) -> None:
    fname = path / f"acc_{neuron_name}.txt"
    lines = [f"{k}: {v}" for k, v in metrics.items()]
    fname.write_text("\n".join(lines))


def plot_accuracies_bar(acc_dict: Dict[str, float], out_png: Path) -> None:
    names = list(acc_dict.keys())
    values = [acc_dict[n] for n in names]
    plt.figure(figsize=(8, 4))
    plt.bar(names, values, color="skyblue")
    plt.ylabel("Accuracy")
    plt.xticks(rotation=45)
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png)
    plt.close()


def plot_accuracies_overlay(acc_dict: Dict[str, float], out_png: Path) -> None:
    names = list(acc_dict.keys())
    values = [acc_dict[n] for n in names]
    plt.figure(figsize=(8, 4))
    plt.plot(names, values, marker="o")
    plt.ylabel("Accuracy")
    plt.xticks(rotation=45)
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png)
    plt.close()
