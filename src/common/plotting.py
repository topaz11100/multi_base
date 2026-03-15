from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def _ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def save_line_plot(
    path: str,
    y_dict: Dict[str, Sequence[float]],
    x: Optional[Sequence[float]] = None,
    title: str = "",
    xlabel: str = "epoch",
    ylabel: str = "",
) -> None:
    _ensure_dir(path)
    plt.figure(figsize=(6.4, 4.2))
    for name, y in y_dict.items():
        y_arr = np.asarray(list(y), dtype=float)
        if x is None:
            x_arr = np.arange(len(y_arr))
        else:
            x_arr = np.asarray(list(x), dtype=float)
        plt.plot(x_arr, y_arr, label=name, linewidth=1.6)
    if title:
        plt.title(title)
    plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    # Readability: enable a light grid (spec requirement).
    plt.grid(True, which="both", alpha=0.28)
    if len(y_dict) > 1:
        plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def save_hist_line(
    path: str,
    values: Sequence[float],
    bins: int = 60,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "count",
) -> None:
    _ensure_dir(path)
    v = np.asarray(list(values), dtype=float)
    plt.figure(figsize=(6.4, 4.2))
    plt.hist(v, bins=bins, histtype="step", linewidth=1.6)
    if title:
        plt.title(title)
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    # Readability: show y-grid for histogram counts.
    plt.grid(True, axis="y", alpha=0.28)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def save_heatmap(
    path: str,
    mat: np.ndarray,
    title: str = "",
    xlabel: str = "neuron (or neuron×branch)",
    ylabel: str = "epoch",
    use_log1p: bool = True,
) -> None:
    _ensure_dir(path)
    m = np.asarray(mat, dtype=float)
    if use_log1p:
        m = np.log1p(m)
    plt.figure(figsize=(7.2, 4.6))
    # Use a masked array so inactive entries (NaN) are visually removed.
    mm = np.ma.masked_invalid(m)
    cmap = plt.get_cmap().copy()
    cmap.set_bad(alpha=0.0)
    im = plt.imshow(mm, aspect="auto", interpolation="nearest", cmap=cmap)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    # Readability: enable a light grid (spec requirement).
    plt.grid(True, which="both", alpha=0.18)
    if title:
        plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
