"""Utility helpers for the multiscale XOR experiments."""
from __future__ import annotations

import json
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Mapping

import matplotlib.pyplot as plt
import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def make_result_dir(exp_name: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = Path("/result") / f"{exp_name}_{timestamp}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def save_json(path: Path, obj: Mapping) -> None:
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def plot_single(neuron_name: str, delays: Iterable[int], accs: Iterable[float], save_path_png: Path) -> None:
    plt.figure(dpi=200)
    plt.plot(list(delays), list(accs), marker="o")
    plt.xlabel("delay (start_time)")
    plt.ylabel("accuracy")
    plt.title(f"{neuron_name.upper()} multiscale XOR")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path_png)
    plt.close()


def plot_overlay(neuron_to_accs: Dict[str, Iterable[float]], delays: Iterable[int], save_path_png: Path) -> None:
    plt.figure(dpi=200)
    x = list(delays)
    for neuron, accs in neuron_to_accs.items():
        plt.plot(x, list(accs), marker="o", label=neuron.upper())
    plt.xlabel("delay (start_time)")
    plt.ylabel("accuracy")
    plt.title("Multiscale XOR neuron comparison")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path_png)
    plt.close()


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def tune_hidden_dim(
    builder,
    target_params: int,
    input_dim: int,
    output_dim: int,
    base_hidden: int,
    search: Iterable[int] | None = None,
    tolerance: float = 0.05,
    device: torch.device | str = "cpu",
) -> tuple[int, int]:
    """Search for a hidden dim whose parameter count matches ``target_params``."""
    if search is None:
        search = range(max(4, base_hidden // 2), max(base_hidden * 2, 256) + 1)

    best_dim = base_hidden
    best_params = None
    best_gap = float("inf")

    for dim in search:
        model = builder(input_dim, dim, output_dim, device)
        params = count_parameters(model)
        gap = abs(params - target_params)
        if gap < best_gap:
            best_gap = gap
            best_dim = dim
            best_params = params
        if target_params > 0 and gap / target_params <= tolerance:
            best_dim = dim
            best_params = params
            break
    assert best_params is not None
    return best_dim, best_params
