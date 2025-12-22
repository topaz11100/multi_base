import os
import random
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Tuple

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


@dataclass
class ExperimentPaths:
    root: str
    timestamp: str

    @property
    def base_dir(self) -> str:
        return os.path.join(self.root, f"experiment_{self.timestamp}")

    def ensure(self) -> str:
        os.makedirs(self.base_dir, exist_ok=True)
        return self.base_dir

    def model_log(self, model_name: str) -> str:
        return os.path.join(self.base_dir, f"log_{model_name}.txt")

    def plot_path(self, model_name: str) -> str:
        return os.path.join(self.base_dir, f"plot_{model_name}.png")

    @property
    def combined_plot(self) -> str:
        return os.path.join(self.base_dir, "plot_combined.png")


class SaveManager:
    def __init__(self, root: str = "result") -> None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.paths = ExperimentPaths(root=root, timestamp=timestamp)
        self.paths.ensure()

    def log_settings(self, settings: str) -> None:
        path = os.path.join(self.paths.base_dir, "settings.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(settings)

    def append_log(self, model_name: str, message: str) -> None:
        path = self.paths.model_log(model_name)
        with open(path, "a", encoding="utf-8") as f:
            f.write(message + "\n")

    def save_plot(self, model_name: str, records: List[Tuple[int, float]]) -> None:
        delays, accs = zip(*records) if records else ([], [])
        plt.figure(dpi=300)
        plt.plot(delays, accs, marker="o")
        plt.xlabel("Delay")
        plt.ylabel("Accuracy")
        plt.title(f"{model_name} Delayed XOR")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(self.paths.plot_path(model_name))
        plt.close()

    def save_combined(self, results: Dict[str, List[Tuple[int, float]]]) -> None:
        plt.figure(dpi=300)
        for name, records in results.items():
            if not records:
                continue
            delays, accs = zip(*records)
            plt.plot(delays, accs, marker="o", label=name.upper())
        plt.xlabel("Delay")
        plt.ylabel("Accuracy")
        plt.title("Delayed XOR comparison")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(self.paths.combined_plot)
        plt.close()

    def describe(self) -> str:
        return self.paths.base_dir


def format_settings(args) -> str:
    lines = ["Experiment settings:"]
    for key, value in sorted(vars(args).items()):
        lines.append(f"- {key}: {value}")
    return "\n".join(lines)


def dry_run(model: torch.nn.Module, batch: torch.Tensor) -> None:
    model.eval()
    with torch.no_grad():
        logits = model(batch)
    print(
        f"Dry run success — device: {next(model.parameters()).device}, "
        f"input: {tuple(batch.shape)}, output: {tuple(logits.shape)}"
    )


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def tune_hidden_dim(
    builder,
    target_params: int,
    input_dim: int,
    output_dim: int,
    base_hidden: int,
    device: torch.device,
    search: Iterable[int] | None = None,
    tolerance: float = 0.1,
) -> int:
    """Find a hidden dimension close to target parameter count."""
    if search is None:
        search = list(range(max(2, base_hidden // 2), base_hidden * 2 + 1))

    best_dim = base_hidden
    best_gap = float("inf")
    for dim in search:
        model = builder(input_dim, dim, output_dim, device=device)
        params = count_parameters(model)
        gap = abs(params - target_params)
        if gap < best_gap:
            best_gap = gap
            best_dim = dim
        if params and abs(params - target_params) / target_params <= tolerance:
            best_dim = dim
            break
    return best_dim
