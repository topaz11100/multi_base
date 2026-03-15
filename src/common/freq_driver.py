from __future__ import annotations

import csv
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tqdm.auto import tqdm

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.common.datasets import (
    get_scifar10_loaders,
    get_shd_loaders,
    get_smnist_loaders,
    get_ssc_loaders,
    visualize_input_sequence,
)
from src.common.fft_analysis import band_edges_to_bin_ranges, bin_spectrum, rfft_freqs, rfft_log_mag
from src.common.model_utils import (
    PARAM_CATEGORIES,
    count_active_parameters,
    count_parameters,
    count_trainable_parameters,
    layer_active_param_breakdown,
)
from src.common.plotting import save_heatmap, save_hist_line, save_line_plot
from src.common.readout import apply_readout
from src.common.snn_builder import SNNConfig, build_layer, _disable_output_spikes_
from src.common.utils import ensure_dir, get_backend_flags, get_device, now_timestamp_seoul, save_json, save_text, set_seed, derive_branch_from_S_max


SIGNALS: Tuple[str, ...] = (
    "dendrite_input",
    "dendrite_state",
    "soma_input",
    "soma_state",
    "output",
)


def _model_name_to_builder(model: str) -> str:
    """Map CLI model names to internal builder names (src/common/snn_builder.py)."""
    m = model.lower().strip()
    if m in (
        "my_dh_snn",
        "my-dh-snn",
        "my-dh",
        "proposed_dh_snn",
        "proposed_dh",
        "dh-snn",
        "dhsnn",
        "dh",
    ):
        return "my-dh-snn"
    if m in (
        "my_r_dh_snn",
        "my-r-dh-snn",
        "my_r_snn",
        "my-r-snn",
        "proposed_r_dh_snn",
        "proposed_r_dh",
        "r-dh-snn",
        "r_snn",
        "r-snn",
        "rsnn",
    ):
        return "my-r-dh-snn"
    if m in (
        "my_d_rf",
        "my-d-rf",
        "my_drf",
        "my-drf",
        "proposed_d_rf",
        "proposed_drf",
        "d-rf",
        "drf",
    ):
        return "my-d-rf"
    raise KeyError(f"Unknown freq_analysis model: {model}")


def _signal_mapping(builder_name: str) -> Dict[str, str]:
    """Return the meaning of each recorded signal key (experiment.md §4)."""
    if builder_name in ("my-d-rf",):
        # my_D_RF_neuron.md
        return {
            "dendrite_input": "I_d[t] : (broadcast) synaptic input current into each branch",
            "dendrite_state": "Re{z_d[t]} : branch resonant state (real part)",
            "soma_input": "H[t] : soma drive = (1/s) * sum_d Re{z_d[t]}",
            "soma_state": "H[t] : soma state (same as soma_input in this implementation)",
            "output": "S[t] : output spike",
        }

    if builder_name in ("my-r-dh-snn",):
        # my_R_DH_SNN_neuron.md
        return {
            "dendrite_input": "O_d[t] : soma-dense input O_soma[t] = W_in O[t] (recorded per neuron; broadcast to all branches internally)",
            "dendrite_state": "i_d[t] : dendrite current state",
            "soma_input": "h[t] : soma input = (1/s) * sum_d (w_d ⊙ i_d[t])",
            "soma_state": "u[t] : soma membrane",
            "output": "o[t] : output spikes",
        }

    # DH (my_DH_SNN_neuron.md)
    return {
        "dendrite_input": "I_d[t] : dendrite synaptic input current",
        "dendrite_state": "i_d[t] : dendrite current state",
        "soma_input": "h[t] : soma input (aggregated dendrite current)",
        "soma_state": "u[t] : soma membrane",
        "output": "o[t] : output spikes",
    }


def _layer_names(hidden: Sequence[int]) -> List[str]:
    return [f"hidden{i+1}" for i in range(len(hidden))] + ["output"]


def _load_dataset(
    dataset: str,
    data_root: str,
    batch_size: int,
    num_workers: int,
    download: bool,
    cifar_mode: str = "parallel",
    T_event: int = 250,
    seed: Optional[int] = None,
):
    """Return train_loader, test_loader, num_classes, input_dim, T, dt_ms."""
    d = dataset.lower()
    if d in ("s-mnist", "smnist"):
        train_loader, test_loader, num_classes, input_dim, T = get_smnist_loaders(
            data_root, batch_size=batch_size, num_workers=num_workers, download=download, seed=seed
        )
        dt_ms = 1.0
    elif d in ("s-cifar10", "scifar10"):
        train_loader, test_loader, num_classes, input_dim, T = get_scifar10_loaders(
            data_root,
            batch_size=batch_size,
            num_workers=num_workers,
            download=download,
            mode=cifar_mode,
            seed=seed,
        )
        dt_ms = 1.0
    elif d in ("shd",):
        train_loader, test_loader, num_classes, input_dim, T = get_shd_loaders(
            data_root, batch_size=batch_size, num_workers=num_workers, download=download, T=T_event, seed=seed
        )
        dt_ms = 1000.0 / float(T)
        # Align meta dt_ms to the dataset's actual discretization if available.
        try:
            ds = getattr(train_loader, "dataset", None)
            # unwrap Subset/Wrapper datasets
            while hasattr(ds, "dataset"):
                ds = getattr(ds, "dataset")
            if ds is not None and hasattr(ds, "dt"):
                dt_ms = 1000.0 * float(getattr(ds, "dt"))
        except Exception:
            pass
    elif d in ("ssc",):
        train_loader, test_loader, num_classes, input_dim, T = get_ssc_loaders(
            data_root, batch_size=batch_size, num_workers=num_workers, download=download, T=T_event, seed=seed
        )
        dt_ms = 1000.0 / float(T)
        # Align meta dt_ms to the dataset's actual discretization if available.
        try:
            ds = getattr(train_loader, "dataset", None)
            # unwrap Subset/Wrapper datasets
            while hasattr(ds, "dataset"):
                ds = getattr(ds, "dataset")
            if ds is not None and hasattr(ds, "dt"):
                dt_ms = 1000.0 * float(getattr(ds, "dt"))
        except Exception:
            pass
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    return train_loader, test_loader, int(num_classes), int(input_dim), int(T), float(dt_ms)


class MultiLayerSNNClassifier(nn.Module):
    """SNN with an arbitrary number of hidden layers.

    Each layer must implement:
      - forward_sequence(x_seq, record: bool) -> y_seq (and optionally rec dict)
      - optional regularization_loss(lambda_ortho, lambda_s)

    Readout:
      - membrane potential only (mean output-layer soma_state over time)
    """

    def __init__(self, layers: Sequence[nn.Module], *, disable_output_spikes: bool = True):
        super().__init__()
        self.layers = nn.ModuleList(list(layers))
        # This classifier uses membrane-potential readout (mean over time of soma_state).
        # If the output layer spikes, reset dynamics can clamp logits. Disable spikes by default.
        if disable_output_spikes and len(self.layers) > 0:
            _disable_output_spikes_(self.layers[-1])

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        h = x_seq
        for li, layer in enumerate(self.layers):
            is_last = (li == len(self.layers) - 1)
            if is_last:
                _, rec = layer.forward_sequence(h, record=("soma_state",))
                return apply_readout(soma_seq=rec.get("soma_state"))
            h = layer.forward_sequence(h, record=False)
        raise RuntimeError("MultiLayerSNNClassifier requires at least one layer")

    def forward_with_records(self, x_seq: torch.Tensor):
        h = x_seq
        recs: List[Dict[str, torch.Tensor]] = []
        for layer in self.layers:
            h, rec = layer.forward_sequence(h, record=True)
            recs.append(rec)
        soma_seq = recs[-1].get("soma_state") if len(recs) > 0 else None
        logits = apply_readout(soma_seq=soma_seq)
        return logits, recs

    def regularization_loss(self, lambda_ortho: float = 0.0, lambda_s: float = 0.0) -> torch.Tensor:
        # NOTE: s-complexity is defined as a *global* mean over all neurons in the model:
        #   L_s = (1/N_total) * sum_over_all_neurons s
        # Therefore, we compute s-regularization at the model level (not as a sum of per-layer means).
        from src.common.model_utils import s_complexity_mean

        loss = None
        for layer in self.layers:
            if hasattr(layer, "regularization_loss") and callable(getattr(layer, "regularization_loss")):
                # Avoid double-counting s: each layer may implement its own lambda_s term.
                l = layer.regularization_loss(lambda_ortho=lambda_ortho, lambda_s=0.0)  # type: ignore
                loss = l if loss is None else (loss + l)
        if loss is None:
            loss = torch.zeros((), device=next(self.parameters()).device)

        if lambda_s != 0.0:
            loss = loss + float(lambda_s) * s_complexity_mean(self)

        return loss


@torch.no_grad()
def _evaluate_classifier(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> Tuple[float, float]:
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        loss_sum += float(loss.item()) * int(x.shape[0])
        pred = logits.argmax(dim=1)
        correct += int((pred == y).sum().item())
        total += int(x.shape[0])
    if total == 0:
        return 0.0, 0.0
    return loss_sum / float(total), correct / float(total)


def _train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    lambda_ortho: float = 0.0,
    lambda_s: float = 0.0,
) -> float:
    model.train()
    total = 0
    loss_sum = 0.0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        if (lambda_ortho != 0.0 or lambda_s != 0.0) and hasattr(model, "regularization_loss"):
            loss = loss + model.regularization_loss(lambda_ortho=lambda_ortho, lambda_s=lambda_s)  # type: ignore
        loss.backward()
        optimizer.step()

        loss_sum += float(loss.item()) * int(x.shape[0])
        total += int(x.shape[0])
    return loss_sum / float(total) if total > 0 else 0.0


@torch.no_grad()
def _select_probe_samples(
    test_loader: DataLoader,
    num_classes: int,
    seed: int,
) -> Dict[int, Dict[str, Any]]:
    """Uniformly sample one test example per label using per-label reservoir sampling.

    Returns dict label -> {dataset_index, label, x (CPU tensor)}.
    Note: dataset_index equals sequential scan index because test_loader uses shuffle=False.
    """
    rng = np.random.RandomState(int(seed))
    counts: Dict[int, int] = {}
    chosen: Dict[int, Dict[str, Any]] = {}
    scan_index = 0
    for x, y in test_loader:
        x_cpu = x.detach().cpu()
        y_cpu = y.detach().cpu()
        B = int(x_cpu.shape[0])
        for i in range(B):
            lbl = int(y_cpu[i].item())
            if lbl < 0 or lbl >= num_classes:
                scan_index += 1
                continue
            c = counts.get(lbl, 0) + 1
            counts[lbl] = c
            # reservoir replace with prob 1/c
            if (lbl not in chosen) or (rng.rand() < (1.0 / float(c))):
                chosen[lbl] = {
                    "dataset_index": int(scan_index),
                    "label": int(lbl),
                    "x": x_cpu[i].clone(),
                }
            scan_index += 1
    # Keep only existing labels
    return {lbl: chosen[lbl] for lbl in sorted(chosen.keys())}


def _ensure_epoch_dir(label_dir: str, epoch: int) -> str:
    return ensure_dir(os.path.join(label_dir, f"epoch{epoch:04d}"))


def _plot_multiline(
    path: str,
    ys: np.ndarray,
    x: Optional[np.ndarray] = None,
    xlabel: str = "",
    ylabel: str = "",
    title: str = "",
    legend_labels: Optional[Sequence[str]] = None,
    legend_prefix: str = "",
) -> None:
    ensure_dir(os.path.dirname(path))
    ys = np.asarray(ys, dtype=float)
    if ys.ndim != 2:
        raise ValueError(f"ys must be 2D, got shape {ys.shape}")
    if x is None:
        x = np.arange(ys.shape[1])
    plt.figure(figsize=(6.8, 3.6))
    for i in range(ys.shape[0]):
        label = None
        if legend_labels is not None:
            if i < len(legend_labels):
                label = str(legend_labels[i])
        elif legend_prefix:
            label = f"{legend_prefix}{i}"
        plt.plot(x, ys[i], linewidth=1.0, label=label)
    if title:
        plt.title(title)
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    # Readability: grid on all multiline plots.
    plt.grid(True, which="both", alpha=0.28)
    if legend_labels is not None or legend_prefix:
        plt.legend(fontsize=7, ncol=2, frameon=False)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def _prep_time_matrix(signal_tensor: torch.Tensor) -> torch.Tensor:
    """Convert a recorded signal tensor to a 2D matrix for FFT/Δ analysis.

    Input is one signal from one layer with shape:
      - (B,T,N,D) for dendrite_* signals
      - (B,T,N)   for soma_* / output
    Output:
      - (N*D, T) for dendrite signals
      - (N, T)   otherwise
    """
    if signal_tensor.ndim == 4:
        s0 = signal_tensor[0]  # (T,N,D)
        T = int(s0.shape[0])
        x_time = s0.permute(1, 2, 0).contiguous().view(-1, T)
        return x_time
    if signal_tensor.ndim == 3:
        s0 = signal_tensor[0]  # (T,N)
        x_time = s0.permute(1, 0).contiguous()  # (N,T)
        return x_time
    raise ValueError(f"Unexpected signal tensor ndim={signal_tensor.ndim}")


def _save_layer_output_heatmaps(
    out_dir: str,
    layer_name: str,
    out_spk: torch.Tensor,
    freqs: np.ndarray,
    fft_band_edges: Optional[Sequence[float]],
    fft_band_ranges: Optional[List[Tuple[int, int]]],
    fft_band_reduce: str,
) -> None:
    """Save time/FFT/binned heatmaps for a layer output spike train (experiment.md §6.6)."""
    ensure_dir(out_dir)

    # out_spk: (T,N)
    spk = out_spk.detach().to(torch.float32).cpu().numpy().T  # (N,T)

    # Use origin="lower" so that neuron index increases upward.
    # (spec: y-axis = neuron index, horizontal line = one neuron's trace)
    plt.figure(figsize=(7.2, 4.6))
    im = plt.imshow(spk, aspect="auto", interpolation="nearest", origin="lower")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xlabel("time (step)")
    plt.ylabel("neuron")
    plt.title(f"{layer_name} output spikes")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{layer_name}_output_time.png"), dpi=200)
    plt.close()

    # FFT heatmap
    spk_t = torch.from_numpy(spk).to(torch.float32)  # (N,T)
    S = rfft_log_mag(spk_t, dim=-1).cpu().numpy()  # (N,F)

    # Align image extent so that pixel centers match the actual rFFT frequency bins.
    # rfftfreq bins are uniformly spaced in normalized frequency (cycles/step).
    if len(freqs) >= 2:
        df = float(freqs[1] - freqs[0])
    else:
        df = 1.0
    x0 = float(freqs[0] - 0.5 * df)
    x1 = float(freqs[-1] + 0.5 * df)
    y0 = -0.5
    y1 = float(S.shape[0] - 0.5)

    plt.figure(figsize=(7.2, 4.6))
    im = plt.imshow(
        S,
        aspect="auto",
        interpolation="nearest",
        extent=[x0, x1, y0, y1],
        origin="lower",
    )
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xlabel("frequency (cycles/step)")
    plt.ylabel("neuron")
    plt.title(f"{layer_name} output rFFT")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{layer_name}_output_fft.png"), dpi=200)
    plt.close()

    # Binned heatmap
    if fft_band_edges is not None and fft_band_ranges is not None and len(fft_band_ranges) > 0:
        S_t = torch.from_numpy(S)
        Sb = bin_spectrum(S_t, fft_band_ranges, dim=-1, reduce=fft_band_reduce).cpu().numpy()  # (N,B)
        x_edges = np.asarray(list(fft_band_edges), dtype=float)
        y_edges = np.arange(Sb.shape[0] + 1, dtype=float)
        plt.figure(figsize=(7.2, 4.6))
        im = plt.pcolormesh(x_edges, y_edges, Sb, shading="auto")
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.xlabel("frequency (cycles/step)")
        plt.ylabel("neuron")
        plt.title(f"{layer_name} output binned rFFT ({fft_band_reduce})")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{layer_name}_output_binned.png"), dpi=200)
        plt.close()


def _save_neuron_analysis(
    neuron_dir: str,
    epoch: int,
    layer_name: str,
    neuron_idx: int,
    layer: nn.Module,
    rec: Dict[str, torch.Tensor],
    delta_hist_exact: Dict[str, List[np.ndarray]],
    delta_hist_binned: Optional[Dict[str, List[np.ndarray]]],
    Dmax: int,
    freqs: np.ndarray,
    fft_band_edges: Optional[Sequence[float]],
    fft_band_ranges: Optional[List[Tuple[int, int]]],
    fft_band_reduce: str,
    mask_hist_layer: Optional[List[np.ndarray]] = None,
) -> None:
    """Save per-neuron time/FFT/Δ plots (experiment.md §6.5).

    NOTE:
      - distribution/ (parameter trajectory) plots are generated **once** at the final epoch
        to avoid duplicated outputs (수정점).
      - dendrite_* plots follow the active-only rule (mask==0 excluded).
    """
    base = ensure_dir(neuron_dir)
    sig_dir = ensure_dir(os.path.join(base, "signals"))
    delta_dir = ensure_dir(os.path.join(base, "delta"))

    # x-axis for Δ plots should reflect epoch numbers (1..epoch)
    epoch_x = np.arange(1, epoch + 1)

    # Optional (Dmax, epoch) mask history for this neuron (active-only rule).
    mask_by_epoch: Optional[np.ndarray] = None
    mask_current: Optional[np.ndarray] = None
    if mask_hist_layer is not None and Dmax > 0:
        try:
            # mask_hist_layer elements are flattened (N*Dmax,)
            idx0 = int(neuron_idx) * int(Dmax)
            mask_by_epoch = np.stack([m[idx0 : idx0 + int(Dmax)] for m in mask_hist_layer[:epoch]], axis=1)
            mask_current = mask_by_epoch[:, -1]
        except Exception:
            mask_by_epoch = None
            mask_current = None

    for sig in SIGNALS:
        s = rec[sig]  # (B,T,N,...) or (B,T,N)
        if s.ndim == 4:
            # (B,T,N,D)
            xt_full = s[0, :, neuron_idx, :].detach().cpu().to(torch.float32).numpy().T  # (D,T)
            T = int(xt_full.shape[1])

            # Active-only: select branches with mask>0 at the current epoch.
            if mask_current is not None:
                active = (mask_current > 0.0)
                if bool(active.any()):
                    xt = xt_full[active, :]
                    branch_ids = np.nonzero(active)[0]
                    legend_labels = [f"d{int(i)}" for i in branch_ids]
                else:
                    xt = xt_full
                    legend_labels = [f"d{int(i)}" for i in range(xt.shape[0])]
            else:
                xt = xt_full
                legend_labels = [f"d{int(i)}" for i in range(xt.shape[0])]

            # time
            _plot_multiline(
                os.path.join(sig_dir, f"{sig}_time.png"),
                xt,
                x=np.arange(T),
                xlabel="time (step)",
                ylabel=sig,
                title=f"{layer_name} neuron{neuron_idx} {sig} time",
                legend_labels=legend_labels,
            )
            # FFT (exact)
            X = torch.from_numpy(xt).to(torch.float32)
            Sx = rfft_log_mag(X, dim=-1).cpu().numpy()  # (D,F)
            _plot_multiline(
                os.path.join(sig_dir, f"{sig}_fft.png"),
                Sx,
                x=freqs,
                xlabel="frequency (cycles/step)",
                ylabel="log(1+|rFFT|)",
                title=f"{layer_name} neuron{neuron_idx} {sig} rFFT",
                legend_labels=legend_labels,
            )
            # FFT (binned)
            if fft_band_edges is not None and fft_band_ranges is not None and len(fft_band_ranges) > 0:
                Sb = bin_spectrum(
                    torch.from_numpy(Sx).to(torch.float32),
                    fft_band_ranges,
                    dim=-1,
                    reduce=fft_band_reduce,
                ).cpu().numpy()  # (D,B)
                centers = np.asarray(
                    [(float(fft_band_edges[i]) + float(fft_band_edges[i + 1])) / 2.0 for i in range(len(fft_band_edges) - 1)],
                    dtype=float,
                )
                _plot_multiline(
                    os.path.join(sig_dir, f"{sig}_fft_band.png"),
                    Sb,
                    x=centers,
                    xlabel="frequency band center (cycles/step)",
                    ylabel=f"binned ({fft_band_reduce}) log(1+|rFFT|)",
                    title=f"{layer_name} neuron{neuron_idx} {sig} binned",
                    legend_labels=legend_labels,
                )

            # Δ (exact) per-branch
            hist = delta_hist_exact[sig]  # list of arrays shape (N*D,)
            idx0 = neuron_idx * Dmax
            branch_delta = np.stack([h[idx0 : idx0 + Dmax] for h in hist], axis=1)  # (Dmax,epoch)
            if mask_by_epoch is not None:
                branch_delta = branch_delta.astype(float, copy=True)
                branch_delta[mask_by_epoch <= 0.0] = np.nan
            _plot_multiline(
                os.path.join(delta_dir, f"{sig}_delta.png"),
                branch_delta,
                x=epoch_x,
                xlabel="epoch",
                ylabel="Δ",
                title=f"{layer_name} neuron{neuron_idx} {sig} Δ(exact)",
                legend_labels=[f"d{int(i)}" for i in range(int(Dmax))],
            )

            # Δ (binned) per-branch
            if delta_hist_binned is not None:
                hist_b = delta_hist_binned[sig]
                branch_delta_b = np.stack([h[idx0 : idx0 + Dmax] for h in hist_b], axis=1)
                if mask_by_epoch is not None:
                    branch_delta_b = branch_delta_b.astype(float, copy=True)
                    branch_delta_b[mask_by_epoch <= 0.0] = np.nan
                _plot_multiline(
                    os.path.join(delta_dir, f"{sig}_binned_delta.png"),
                    branch_delta_b,
                    x=epoch_x,
                    xlabel="epoch",
                    ylabel="Δ",
                    title=f"{layer_name} neuron{neuron_idx} {sig} Δ(binned)",
                    legend_labels=[f"d{int(i)}" for i in range(int(Dmax))],
                )

        elif s.ndim == 3:
            # (B,T,N)
            xt = s[0, :, neuron_idx].detach().cpu().to(torch.float32).numpy()  # (T,)
            T = int(xt.shape[0])
            # time
            save_line_plot(
                os.path.join(sig_dir, f"{sig}_time.png"),
                {sig: xt},
                x=np.arange(T),
                xlabel="time (step)",
                ylabel=sig,
                title=f"{layer_name} neuron{neuron_idx} {sig} time",
            )
            # FFT
            Sx = rfft_log_mag(xt, dim=-1)
            save_line_plot(
                os.path.join(sig_dir, f"{sig}_fft.png"),
                {sig: np.asarray(Sx, dtype=float)},
                x=freqs,
                xlabel="frequency (cycles/step)",
                ylabel="log(1+|rFFT|)",
                title=f"{layer_name} neuron{neuron_idx} {sig} rFFT",
            )
            if fft_band_edges is not None and fft_band_ranges is not None and len(fft_band_ranges) > 0:
                Sb = bin_spectrum(np.asarray(Sx, dtype=float), fft_band_ranges, dim=-1, reduce=fft_band_reduce)
                centers = np.asarray(
                    [(float(fft_band_edges[i]) + float(fft_band_edges[i + 1])) / 2.0 for i in range(len(fft_band_edges) - 1)],
                    dtype=float,
                )
                save_line_plot(
                    os.path.join(sig_dir, f"{sig}_fft_band.png"),
                    {sig: np.asarray(Sb, dtype=float)},
                    x=centers,
                    xlabel="frequency band center (cycles/step)",
                    ylabel=f"binned ({fft_band_reduce}) log(1+|rFFT|)",
                    title=f"{layer_name} neuron{neuron_idx} {sig} binned",
                )

            # Δ
            hist = np.asarray([h[neuron_idx] for h in delta_hist_exact[sig]], dtype=float)
            save_line_plot(
                os.path.join(delta_dir, f"{sig}_delta.png"),
                {sig: hist},
                x=epoch_x,
                xlabel="epoch",
                ylabel="Δ",
                title=f"{layer_name} neuron{neuron_idx} {sig} Δ(exact)",
            )
            if delta_hist_binned is not None:
                hist_b = np.asarray([h[neuron_idx] for h in delta_hist_binned[sig]], dtype=float)
                save_line_plot(
                    os.path.join(delta_dir, f"{sig}_binned_delta.png"),
                    {sig: hist_b},
                    x=epoch_x,
                    xlabel="epoch",
                    ylabel="Δ",
                    title=f"{layer_name} neuron{neuron_idx} {sig} Δ(binned)",
                )
        else:
            raise ValueError(f"Unexpected ndim for {sig}: {s.ndim}")

    # (distribution plots removed here; generated once at final epoch)


def _save_neuron_param_trajectories(
    dist_dir: str,
    layer_name: str,
    neuron_idx: int,
    epoch_x: np.ndarray,
    param_hist: Dict[str, List[Any]],
    Dmax: int,
    mask_hist_layer: Optional[List[np.ndarray]] = None,
) -> None:
    """Save per-neuron *parameter* trajectories across epochs.

    This corresponds to experiment.md §6.5 `distribution/`, but the content is
    a trajectory plot (epoch -> value) rather than per-epoch histograms (수정점).
    """

    ensure_dir(dist_dir)

    # Prepare branch mask history for this neuron, if available.
    mask_by_epoch: Optional[np.ndarray] = None
    if mask_hist_layer is not None and int(Dmax) > 0:
        try:
            idx0 = int(neuron_idx) * int(Dmax)
            mask_by_epoch = np.stack([m[idx0 : idx0 + int(Dmax)] for m in mask_hist_layer[: len(epoch_x)]], axis=1)
        except Exception:
            mask_by_epoch = None

    def _mask_branches(mat: np.ndarray) -> np.ndarray:
        out = np.asarray(mat, dtype=float).copy()
        if mask_by_epoch is not None:
            out[mask_by_epoch <= 0.0] = np.nan
        return out

    # --- structure ---
    if "D_int" in param_hist and len(param_hist["D_int"]) == len(epoch_x):
        save_line_plot(
            os.path.join(dist_dir, "D_int.png"),
            {"D_int": np.asarray(param_hist["D_int"], dtype=float)},
            x=epoch_x,
            xlabel="epoch",
            ylabel="D_int",
            title=f"{layer_name} neuron{neuron_idx} D_int trajectory",
        )

    if "s" in param_hist and len(param_hist["s"]) == len(epoch_x):
        save_line_plot(
            os.path.join(dist_dir, "s.png"),
            {"s": np.asarray(param_hist["s"], dtype=float)},
            x=epoch_x,
            xlabel="epoch",
            ylabel="s",
            title=f"{layer_name} neuron{neuron_idx} s trajectory",
        )

    # --- timing / resonance ---
    if "alpha" in param_hist and len(param_hist["alpha"]) == len(epoch_x):
        alpha_mat = np.stack([np.asarray(v, dtype=float).reshape(-1)[: int(Dmax)] for v in param_hist["alpha"]], axis=1)
        alpha_mat = _mask_branches(alpha_mat)
        _plot_multiline(
            os.path.join(dist_dir, "alpha.png"),
            alpha_mat,
            x=epoch_x,
            xlabel="epoch",
            ylabel="alpha",
            title=f"{layer_name} neuron{neuron_idx} alpha trajectory",
            legend_labels=[f"d{int(i)}" for i in range(int(Dmax))],
        )

    if "beta" in param_hist and len(param_hist["beta"]) == len(epoch_x):
        save_line_plot(
            os.path.join(dist_dir, "beta.png"),
            {"beta": np.asarray(param_hist["beta"], dtype=float)},
            x=epoch_x,
            xlabel="epoch",
            ylabel="beta",
            title=f"{layer_name} neuron{neuron_idx} beta trajectory",
        )

    if "tau" in param_hist and len(param_hist["tau"]) == len(epoch_x):
        tau_mat = np.stack([np.asarray(v, dtype=float).reshape(-1)[: int(Dmax)] for v in param_hist["tau"]], axis=1)
        tau_mat = _mask_branches(tau_mat)
        _plot_multiline(
            os.path.join(dist_dir, "tau.png"),
            tau_mat,
            x=epoch_x,
            xlabel="epoch",
            ylabel="tau",
            title=f"{layer_name} neuron{neuron_idx} tau trajectory",
            legend_labels=[f"d{int(i)}" for i in range(int(Dmax))],
        )

    if "omega" in param_hist and len(param_hist["omega"]) == len(epoch_x):
        om_mat = np.stack([np.asarray(v, dtype=float).reshape(-1)[: int(Dmax)] for v in param_hist["omega"]], axis=1)
        om_mat = _mask_branches(om_mat)
        _plot_multiline(
            os.path.join(dist_dir, "omega.png"),
            om_mat,
            x=epoch_x,
            xlabel="epoch",
            ylabel="omega",
            title=f"{layer_name} neuron{neuron_idx} omega trajectory",
            legend_labels=[f"d{int(i)}" for i in range(int(Dmax))],
        )





def _save_hist_overlay_lines(
    out_path: str,
    values_by_epoch: Sequence[np.ndarray],
    epoch_x: np.ndarray,
    bins: int = 60,
    title: str = "",
    xlabel: str = "weight",
    density: bool = True,
) -> None:
    """Overlay per-epoch histograms as multiple lines in a single plot.

    Used for `label_*/epochXXXX/hiddenK_neuron_*/weight/*.png`.

    Notes:
      - We use a shared binning (global min/max across epochs) so curves are comparable.
      - Legend is limited to a small subset of epochs to keep the plot readable.
    """
    ensure_dir(os.path.dirname(out_path))

    # Determine global range.
    vmin = float("inf")
    vmax = float("-inf")
    any_data = False
    for v in values_by_epoch:
        vv = np.asarray(v, dtype=float).reshape(-1)
        if vv.size == 0:
            continue
        any_data = True
        vmin = min(vmin, float(np.nanmin(vv)))
        vmax = max(vmax, float(np.nanmax(vv)))

    if not any_data:
        return

    if not np.isfinite(vmin) or not np.isfinite(vmax):
        return

    if abs(vmax - vmin) < 1e-12:
        vmin -= 1e-6
        vmax += 1e-6

    edges = np.linspace(vmin, vmax, int(bins) + 1, dtype=float)
    centers = 0.5 * (edges[:-1] + edges[1:])

    plt.figure(figsize=(6.6, 4.3))

    n = int(len(values_by_epoch))
    # Show legend only for a small set of representative epochs.
    max_labels = 12
    if n <= max_labels:
        label_indices = set(range(n))
    else:
        label_indices = set(np.unique(np.linspace(0, n - 1, max_labels, dtype=int)).tolist())

    any_label = False
    for i, v in enumerate(values_by_epoch):
        vv = np.asarray(v, dtype=float).reshape(-1)
        if vv.size == 0:
            continue
        hist, _ = np.histogram(vv, bins=edges, density=bool(density))
        label = None
        if i in label_indices:
            label = f"epoch{int(epoch_x[i])}"
            any_label = True
        plt.plot(centers, hist, linewidth=1.2, alpha=0.65, label=label)

    if title:
        plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("density" if density else "count")
    plt.grid(True, axis="y", alpha=0.28)
    if any_label:
        plt.legend(frameon=False, fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _active_weights_over_epochs(
    weight_hist: Sequence[np.ndarray],
    neuron_idx: int,
    Dmax: int,
    mask_hist_layer: Optional[List[np.ndarray]] = None,
) -> List[np.ndarray]:
    """Convert raw weight snapshots into a list of 1D arrays (active-only) per epoch."""
    out: List[np.ndarray] = []
    idx0 = int(neuron_idx) * int(Dmax)
    for ep, w in enumerate(weight_hist):
        ww = np.asarray(w)
        if ww.ndim == 2:
            # (D,in_dim) : my_DH per-branch synapse weights
            if mask_hist_layer is not None and ep < len(mask_hist_layer) and ww.shape[0] == int(Dmax):
                m = np.asarray(mask_hist_layer[ep], dtype=float).reshape(-1)[idx0 : idx0 + int(Dmax)]
                active = m > 0.0
                out.append(ww[active, :].reshape(-1))
            else:
                out.append(ww.reshape(-1))
        else:
            # 1D : W_mix row or fc row
            if mask_hist_layer is not None and ep < len(mask_hist_layer) and ww.size == int(Dmax):
                m = np.asarray(mask_hist_layer[ep], dtype=float).reshape(-1)[idx0 : idx0 + int(Dmax)]
                active = m > 0.0
                out.append(ww.reshape(-1)[active])
            else:
                out.append(ww.reshape(-1))
    return out



def _extract_layer_timing_and_structure(layer: nn.Module) -> Dict[str, Any]:
    """Extract structure + timing parameters for distribution plots.

    For proposed *variable-branch* models, many timing parameters are branch-wise
    (shape: (N, Dmax)). The experiment spec emphasizes *active-only* artifacts,
    so we filter branch-wise timing parameters (alpha/tau/omega) to include only
    branches with mask>0 at the current epoch.

    For non-dendritic baselines (no soft_mask), values are returned as-is.
    """
    out: Dict[str, Any] = {}

    # structure params (s, D_int, ...)
    if hasattr(layer, 'get_structure_params') and callable(getattr(layer, 'get_structure_params')):
        try:
            out.update(layer.get_structure_params())  # type: ignore
        except Exception:
            pass

    # timing params
    tp = None
    if hasattr(layer, 'get_timing_params') and callable(getattr(layer, 'get_timing_params')):
        try:
            tp = layer.get_timing_params()  # type: ignore
        except Exception:
            tp = None

    if not tp:
        return out

    # Active-only filter mask for branch-wise params
    mask = None
    if hasattr(layer, 'soft_mask') and callable(getattr(layer, 'soft_mask')):
        try:
            m = layer.soft_mask(torch.float32)  # type: ignore
            if torch.is_tensor(m):
                mask = m.detach().cpu().numpy()
        except Exception:
            mask = None

    N = None
    D = None
    if mask is not None:
        try:
            N = int(getattr(layer, 'output_dim'))
            D = int(getattr(layer, 'branch'))
        except Exception:
            N, D = None, None

    for k, v in tp.items():
        # Convert to numpy for consistent downstream handling
        try:
            if torch.is_tensor(v):
                arr = v.detach().cpu().numpy().reshape(-1)
            else:
                import numpy as np
                arr = np.asarray(v, dtype=float).reshape(-1)
        except Exception:
            continue

        if k in ('alpha', 'tau', 'omega') and mask is not None and N is not None and D is not None:
            try:
                import numpy as np
                if mask.shape == (N, D) and arr.size == (N * D):
                    arr2 = arr.reshape(N, D)
                    arr = arr2[mask > 0.0].reshape(-1)
            except Exception:
                pass

        out[k] = arr

    return out



def _extract_layer_weight_active(layer: nn.Module) -> np.ndarray:
    """Return active-only weight vector for distribution plots.

    Spec requirement (experiment.md + 수정점):
      - Do NOT plot weights for inactive (masked) branches.
      - Remove any *_weight_all plots entirely.
    """
    # my_DH: W has shape (N*Dmax, in_dim)
    if hasattr(layer, "W") and hasattr(layer, "d_int"):
        W_mat = getattr(layer, "W").detach().cpu().numpy()
        d = layer.d_int()  # type: ignore
        d_vec = d.detach().cpu().to(torch.int64).numpy().reshape(-1) if torch.is_tensor(d) else np.asarray(d, dtype=np.int64).reshape(-1)
        Dmax = int(getattr(layer, "branch"))
        out_dim = int(getattr(layer, "output_dim"))
        in_dim = int(getattr(layer, "input_dim"))
        W_mat = W_mat.reshape(out_dim * Dmax, in_dim)
        active_rows: List[int] = []
        for n in range(out_dim):
            dn = int(d_vec[n]) if n < d_vec.shape[0] else 0
            active_rows.extend(list(range(n * Dmax, n * Dmax + dn)))
        W_active = W_mat[np.asarray(active_rows, dtype=np.int64), :].reshape(-1)
        return W_active

    # my_R_DH: W_mix is (N,Dmax)
    if hasattr(layer, "W_mix") and hasattr(layer, "d_int"):
        Wm = getattr(layer, "W_mix").detach().cpu().numpy()
        d = layer.d_int()  # type: ignore
        d_vec = d.detach().cpu().to(torch.int64).numpy().reshape(-1) if torch.is_tensor(d) else np.asarray(d, dtype=np.int64).reshape(-1)
        active_rows = []
        for n in range(Wm.shape[0]):
            dn = int(d_vec[n]) if n < d_vec.shape[0] else 0
            if dn > 0:
                active_rows.append(Wm[n, :dn].reshape(-1))
        w_parts: List[np.ndarray] = []
        # include axon-shared input weights if present (always active)
        if hasattr(layer, "W_in"):
            try:
                w_in = getattr(layer, "W_in").detach().cpu().numpy().reshape(-1)
                w_parts.append(np.asarray(w_in, dtype=float))
            except Exception:
                pass
        W_active = np.concatenate(active_rows, axis=0) if len(active_rows) > 0 else np.zeros((0,), dtype=Wm.dtype)
        w_parts.append(np.asarray(W_active, dtype=float))
        return np.concatenate(w_parts, axis=0) if len(w_parts) > 1 else w_parts[0]

    # my_D_RF: fc weights always active
    if hasattr(layer, "fc") and isinstance(getattr(layer, "fc"), nn.Linear):
        fc: nn.Linear = getattr(layer, "fc")
        w_parts = [fc.weight.detach().cpu().numpy().reshape(-1)]
        if fc.bias is not None:
            w_parts.append(fc.bias.detach().cpu().numpy().reshape(-1))
        return np.concatenate(w_parts, axis=0)

    # fallback
    ws = []
    for p in layer.parameters():
        if p.requires_grad:
            ws.append(p.detach().cpu().reshape(-1).numpy())
    if not ws:
        z = np.zeros((0,), dtype=np.float32)
        return z
    allw = np.concatenate(ws, axis=0)
    return allw


def _save_distribution_plots(
    out_dir: str,
    model: MultiLayerSNNClassifier,
    epoch: int,
    layer_names: Sequence[str],
    fft_model_tag: str,
) -> None:
    """Save timing/weight distribution plots (experiment.md §6.2)."""
    dist_root = ensure_dir(os.path.join(out_dir, "distribution", f"epoch{epoch:04d}"))
    timing_dir = ensure_dir(os.path.join(dist_root, "timing"))
    weight_dir = ensure_dir(os.path.join(dist_root, "weights"))

    # aggregation
    agg_D_int: List[np.ndarray] = []
    agg_s: List[np.ndarray] = []
    agg_alpha: List[np.ndarray] = []
    agg_beta: List[np.ndarray] = []
    agg_tau: List[np.ndarray] = []
    agg_omega: List[np.ndarray] = []
    agg_w_act: List[np.ndarray] = []

    for lname, layer in zip(layer_names, model.layers):
        prefix = f"layer_{lname}"
        info = _extract_layer_timing_and_structure(layer)

        # structure
        if "D_int" in info:
            v = info["D_int"]
            arr = v.detach().cpu().numpy().reshape(-1) if torch.is_tensor(v) else np.asarray(v, dtype=float).reshape(-1)
            agg_D_int.append(arr)
            save_hist_line(os.path.join(timing_dir, f"{prefix}_D_int.png"), arr, xlabel="D_int")
        if "s" in info:
            v = info["s"]
            arr = v.detach().cpu().numpy().reshape(-1) if torch.is_tensor(v) else np.asarray(v, dtype=float).reshape(-1)
            agg_s.append(arr)
            save_hist_line(os.path.join(timing_dir, f"{prefix}_s.png"), arr, xlabel="s")

        # timing params
        if "alpha" in info and "beta" in info:
            alpha = info["alpha"]
            beta = info["beta"]
            if torch.is_tensor(alpha):
                alpha = alpha.detach().cpu().numpy()
            if torch.is_tensor(beta):
                beta = beta.detach().cpu().numpy()
            alpha = np.asarray(alpha, dtype=float).reshape(-1)
            beta = np.asarray(beta, dtype=float).reshape(-1)
            save_hist_line(os.path.join(timing_dir, f"{prefix}_alpha.png"), alpha, xlabel="alpha")
            save_hist_line(os.path.join(timing_dir, f"{prefix}_beta.png"), beta, xlabel="beta")
            agg_alpha.append(alpha)
            agg_beta.append(beta)

        if "tau" in info and "omega" in info:
            tau = info["tau"]
            omega = info["omega"]
            if torch.is_tensor(tau):
                tau = tau.detach().cpu().numpy()
            if torch.is_tensor(omega):
                omega = omega.detach().cpu().numpy()
            tau = np.asarray(tau, dtype=float).reshape(-1)
            omega = np.asarray(omega, dtype=float).reshape(-1)
            save_hist_line(os.path.join(timing_dir, f"{prefix}_tau.png"), tau, xlabel="tau")
            save_hist_line(os.path.join(timing_dir, f"{prefix}_omega.png"), omega, xlabel="omega")
            agg_tau.append(tau)
            agg_omega.append(omega)

        # weights (active-only)
        w_act = _extract_layer_weight_active(layer)
        save_hist_line(os.path.join(weight_dir, f"{prefix}_weight_active.png"), w_act, xlabel="weight(active)")
        agg_w_act.append(np.asarray(w_act, dtype=float).reshape(-1))

    # model-level
    if agg_D_int:
        save_hist_line(os.path.join(timing_dir, "model_D_int.png"), np.concatenate(agg_D_int, axis=0), xlabel="D_int")
    if agg_s:
        save_hist_line(os.path.join(timing_dir, "model_s.png"), np.concatenate(agg_s, axis=0), xlabel="s")
    if agg_alpha:
        save_hist_line(os.path.join(timing_dir, "model_alpha.png"), np.concatenate(agg_alpha, axis=0), xlabel="alpha")
        save_hist_line(os.path.join(timing_dir, "model_beta.png"), np.concatenate(agg_beta, axis=0), xlabel="beta")
    if agg_tau:
        save_hist_line(os.path.join(timing_dir, "model_tau.png"), np.concatenate(agg_tau, axis=0), xlabel="tau")
        save_hist_line(os.path.join(timing_dir, "model_omega.png"), np.concatenate(agg_omega, axis=0), xlabel="omega")
    if agg_w_act:
        save_hist_line(
            os.path.join(weight_dir, "model_weight_active.png"),
            np.concatenate(agg_w_act, axis=0),
            xlabel="weight(active)",
        )




# ---------------------------------------------------------------------
# R-DH transfer analysis (dendrite_input -> soma_input)
#   - only enabled for builder_name == "my-r-dh-snn"
#   - design doc: paper/proposed/r_dh_snn_transfer_analysis.md
# ---------------------------------------------------------------------

_RDH_CLASS_NAMES: Tuple[str, ...] = ("LP", "BP", "HP", "mixed")


def _rdh_cutoff_cycles_per_step(alpha: np.ndarray) -> np.ndarray:
    """Compute the -3 dB cutoff frequency for the EMA low-pass branch.

    alpha is in (0,1). The closed-form follows paper/proposed/varidble_dendric.md.

    Returns:
      f_c in normalized frequency (cycles/step), in [0, 0.5].
    """
    a = np.asarray(alpha, dtype=float)
    a = np.clip(a, 1e-6, 1.0 - 1e-6)
    # cos(omega_c) = (4a - a^2 - 1) / (2a)
    cos_w = (4.0 * a - a * a - 1.0) / (2.0 * a)
    cos_w = np.clip(cos_w, -1.0, 1.0)
    omega_c = np.arccos(cos_w)  # rad
    f_c = omega_c / (2.0 * np.pi)
    return np.clip(f_c, 0.0, 0.5)


def _rdh_transfer_layer_metrics(
    layer: nn.Module,
    freqs: np.ndarray,
    *,
    eps: float = 1e-12,
    thresholds: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Compute R-DH transfer response metrics for one layer.

    Implements the LTI approximation from paper/proposed/r_dh_snn_transfer_analysis.md:

      H_total,m(e^{jω}) = (1/s_m) * sum_d ( G_{m,d}(s_m) * w_{m,d} * H_{m,d}(e^{jω}) )

    where H_{m,d} is the EMA low-pass:

      H_{m,d}(z) = (1-α_{m,d}) / (1-α_{m,d} z^{-1})

    Frequency axis uses normalized frequency f in [0,0.5] cycles/step, ω = 2π f.

    Returns dict with:
      - per-neuron arrays: f_peak, r0, rpi, bandwidth, class_id
      - summary dict: counts/fractions + means
      - magnitude matrices: A, A_norm (N,F)
      - internal arrays: alpha, gate, w_mix, s, D_int
    """

    # Default classification thresholds (same as doc, converted to cycles/step)
    th = {
        "lp_peak_max": 0.05,   # 0.1π -> 0.05 cycles/step
        "hp_peak_min": 0.45,   # 0.9π -> 0.45 cycles/step
        "lp_r0_min": 0.7,
        "lp_rpi_max": 0.5,
        "hp_r0_max": 0.3,
        "hp_rpi_min": 0.7,
        "bp_r0_max": 0.5,
        "bp_rpi_max": 0.5,
        "db3_thresh": float(2.0 ** (-0.5)),
    }
    if thresholds:
        th.update({str(k): float(v) for k, v in thresholds.items()})

    # Extract required params.
    if not hasattr(layer, "alpha_branch") or not callable(getattr(layer, "alpha_branch")):
        raise ValueError("R-DH transfer analysis requires layer.alpha_branch()")
    if not hasattr(layer, "W_mix"):
        raise ValueError("R-DH transfer analysis requires layer.W_mix")
    if not hasattr(layer, "s") or not callable(getattr(layer, "s")):
        raise ValueError("R-DH transfer analysis requires layer.s()")

    alpha = layer.alpha_branch().detach().cpu().to(torch.float32).numpy()  # type: ignore  # (N,D)
    w_mix = getattr(layer, "W_mix").detach().cpu().to(torch.float32).numpy()  # type: ignore  # (N,D)
    s = layer.s().detach().cpu().to(torch.float32).numpy().reshape(-1)  # type: ignore  # (N,)

    if hasattr(layer, "d_int") and callable(getattr(layer, "d_int")):
        D_int = layer.d_int().detach().cpu().to(torch.int64).numpy().reshape(-1)  # type: ignore
    else:
        D_int = np.full((alpha.shape[0],), int(alpha.shape[1]), dtype=np.int64)

    if hasattr(layer, "soft_mask") and callable(getattr(layer, "soft_mask")):
        gate = layer.soft_mask(torch.float32).detach().cpu().numpy()  # type: ignore  # (N,D)
    else:
        gate = np.ones_like(alpha, dtype=np.float32)

    # Sanity shapes
    if alpha.ndim != 2 or w_mix.ndim != 2 or alpha.shape != w_mix.shape:
        raise ValueError(f"alpha and W_mix must be (N,D) with same shape (got {alpha.shape}, {w_mix.shape})")
    if gate.shape != alpha.shape:
        raise ValueError(f"mask gate must be (N,D) (got {gate.shape}, expected {alpha.shape})")

    N, D = int(alpha.shape[0]), int(alpha.shape[1])
    F = int(np.asarray(freqs).size)
    freqs = np.asarray(freqs, dtype=float).reshape(-1)

    s_safe = np.clip(s.astype(np.float32), 1e-6, None)  # (N,)

    # e^{-jω} with ω = 2π f
    E = np.exp(-1j * 2.0 * np.pi * freqs).astype(np.complex64)  # (F,)

    # Complex transfer per neuron
    H_total = np.zeros((N, F), dtype=np.complex64)

    # Accumulate branch contributions to avoid allocating (N,D,F)
    for d in range(D):
        a = np.clip(alpha[:, d].astype(np.float32), 1e-6, 1.0 - 1e-6)  # (N,)
        coef = (gate[:, d].astype(np.float32) * w_mix[:, d].astype(np.float32)) / s_safe  # (N,)
        # H_d(e^{jω}) = (1-a) / (1 - a e^{-jω})
        Hd = (1.0 - a)[:, None] / (1.0 - a[:, None] * E[None, :])
        H_total += (coef[:, None].astype(np.complex64) * Hd.astype(np.complex64))

    A = np.abs(H_total).astype(np.float32)  # (N,F)
    A_max = np.max(A, axis=1, keepdims=True)
    A_norm = A / (A_max + float(eps))

    # Per-neuron metrics
    r0 = A_norm[:, 0]
    rpi = A_norm[:, -1]

    peak_idx = np.argmax(A_norm, axis=1)
    f_peak = freqs[peak_idx]

    # -3 dB bandwidth around the peak (based on normalized magnitude)
    mask3 = (A_norm >= float(th["db3_thresh"]))
    first_idx = np.argmax(mask3, axis=1)
    last_idx = (F - 1) - np.argmax(mask3[:, ::-1], axis=1)
    bandwidth = freqs[last_idx] - freqs[first_idx]

    # Classification
    lp = (f_peak <= float(th["lp_peak_max"])) & (r0 >= float(th["lp_r0_min"])) & (rpi <= float(th["lp_rpi_max"]))
    hp = (f_peak >= float(th["hp_peak_min"])) & (r0 <= float(th["hp_r0_max"])) & (rpi >= float(th["hp_rpi_min"]))
    bp = (~lp) & (~hp) & (f_peak > float(th["lp_peak_max"])) & (f_peak < float(th["hp_peak_min"])) & (r0 <= float(th["bp_r0_max"])) & (rpi <= float(th["bp_rpi_max"]))

    class_id = np.full((N,), 3, dtype=np.int64)  # mixed
    class_id[lp] = 0
    class_id[bp] = 1
    class_id[hp] = 2

    counts = {
        "LP": int(np.sum(class_id == 0)),
        "BP": int(np.sum(class_id == 1)),
        "HP": int(np.sum(class_id == 2)),
        "mixed": int(np.sum(class_id == 3)),
    }

    summary = {
        "N": int(N),
        "D": int(D),
        "freq_axis": {
            "unit": "cycles/step",
            "F": int(F),
            "f_min": float(freqs[0]) if F > 0 else 0.0,
            "f_max": float(freqs[-1]) if F > 0 else 0.5,
        },
        "thresholds": {k: float(v) for k, v in th.items()},
        "counts": counts,
        "fractions": {k: float(counts[k]) / float(N) if N > 0 else 0.0 for k in counts},
        "means": {
            "f_peak": float(np.mean(f_peak)) if N > 0 else 0.0,
            "r0": float(np.mean(r0)) if N > 0 else 0.0,
            "rpi": float(np.mean(rpi)) if N > 0 else 0.0,
            "bandwidth": float(np.mean(bandwidth)) if N > 0 else 0.0,
            "s": float(np.mean(s_safe)) if N > 0 else 0.0,
            "D_int": float(np.mean(D_int.astype(float))) if N > 0 else 0.0,
        },
    }

    return {
        "alpha": alpha,
        "gate": gate,
        "w_mix": w_mix,
        "s": s_safe,
        "D_int": D_int,
        "H_total": H_total,
        "A": A,
        "A_norm": A_norm,
        "f_peak": f_peak,
        "r0": r0,
        "rpi": rpi,
        "bandwidth": bandwidth,
        "class_id": class_id,
        "summary": summary,
    }


def _save_rdh_transfer_selected_neuron_plots(
    out_dir: str,
    layer_name: str,
    neuron_idx: int,
    freqs: np.ndarray,
    alpha_row: np.ndarray,
    gate_row: np.ndarray,
    w_row: np.ndarray,
    s_val: float,
    A_row: np.ndarray,
    A_norm_row: np.ndarray,
    summary_row: Dict[str, float],
) -> None:
    """Save per-neuron transfer plots (branch cutoff/response + total response)."""
    ensure_dir(out_dir)

    freqs = np.asarray(freqs, dtype=float).reshape(-1)

    # active branches
    gate_row = np.asarray(gate_row, dtype=float).reshape(-1)
    active = gate_row > 0.0
    branch_ids = np.nonzero(active)[0]

    # branch cutoff
    if branch_ids.size > 0:
        fc = _rdh_cutoff_cycles_per_step(np.asarray(alpha_row, dtype=float).reshape(-1)[active])
        save_line_plot(
            os.path.join(out_dir, "branch_cutoff.png"),
            {"cutoff": fc},
            x=[float(i) for i in branch_ids],
            xlabel="branch",
            ylabel="f_c (cycles/step)",
            title=f"{layer_name} neuron{int(neuron_idx)} branch cutoff (active)",
        )
    else:
        # fallback: empty plot
        save_line_plot(
            os.path.join(out_dir, "branch_cutoff.png"),
            {"cutoff": []},
            x=[],
            xlabel="branch",
            ylabel="f_c (cycles/step)",
            title=f"{layer_name} neuron{int(neuron_idx)} branch cutoff (active)",
        )

    # branch magnitude responses
    # NOTE: magnitude responses should be based on complex H_d, then abs.
    if branch_ids.size > 0:
        E = np.exp(-1j * 2.0 * np.pi * freqs).astype(np.complex64)
        mags = []
        labels = []
        for j, d in enumerate(branch_ids.tolist()):
            a = float(np.clip(alpha_row[d], 1e-6, 1.0 - 1e-6))
            Hd = (1.0 - a) / (1.0 - a * E)
            mags.append(np.abs(Hd).astype(float))
            labels.append(f"d{int(d)}")

        # Use the internal multi-line plot helper for readability.
        _plot_multiline(
            os.path.join(out_dir, "branch_response.png"),
            np.stack(mags, axis=0),
            x=freqs,
            xlabel="frequency (cycles/step)",
            ylabel="|H_d|",
            title=f"{layer_name} neuron{int(neuron_idx)} branch responses (active)",
            legend_labels=labels,
        )
    else:
        save_line_plot(
            os.path.join(out_dir, "branch_response.png"),
            {"|H_d|": []},
            x=[],
            xlabel="frequency (cycles/step)",
            ylabel="|H_d|",
            title=f"{layer_name} neuron{int(neuron_idx)} branch responses (active)",
        )

    # total response (unnormalized + normalized)
    save_line_plot(
        os.path.join(out_dir, "total_response.png"),
        {"|H_total|": np.asarray(A_row, dtype=float).reshape(-1)},
        x=freqs,
        xlabel="frequency (cycles/step)",
        ylabel="|H_total|",
        title=(
            f"{layer_name} neuron{int(neuron_idx)} total response |H| (s={float(s_val):.3f})"
        ),
    )

    save_line_plot(
        os.path.join(out_dir, "total_response_norm.png"),
        {"|H_total|/max": np.asarray(A_norm_row, dtype=float).reshape(-1)},
        x=freqs,
        xlabel="frequency (cycles/step)",
        ylabel="normalized magnitude",
        title=(
            f"{layer_name} neuron{int(neuron_idx)} normalized response "
            f"(r0={float(summary_row.get('r0', 0.0)):.3f}, rpi={float(summary_row.get('rpi', 0.0)):.3f}, f_peak={float(summary_row.get('f_peak', 0.0)):.3f})"
        ),
    )


def _save_rdh_transfer_epoch(
    out_dir: str,
    model: MultiLayerSNNClassifier,
    epoch: int,
    layer_names: Sequence[str],
    freqs: np.ndarray,
    analysis_neuron_indices: Optional[List[List[int]]],
    plot_every: int,
    transfer_history: Dict[str, List[Dict[str, Any]]],
    thresholds: Optional[Dict[str, float]] = None,
) -> None:
    """Compute + save R-DH transfer analysis artifacts for one epoch."""

    base = ensure_dir(os.path.join(out_dir, "transfer", f"epoch{int(epoch):04d}"))

    # Per-layer
    for li, (lname, layer) in enumerate(zip(layer_names, model.layers)):
        metrics = _rdh_transfer_layer_metrics(layer, freqs, thresholds=thresholds)

        # summary json (always per epoch)
        summ = dict(metrics["summary"])
        summ.update({"epoch": int(epoch), "layer": str(lname)})
        save_json(os.path.join(base, f"{lname}_summary.json"), summ)
        transfer_history.setdefault(lname, []).append(summ)

        # Plots + selected-neuron visuals only at plot_every epochs
        if int(plot_every) > 0 and (int(epoch) % int(plot_every) == 0):
            # histograms
            save_hist_line(
                os.path.join(base, f"{lname}_hist_f_peak.png"),
                metrics["f_peak"].reshape(-1),
                xlabel="f_peak (cycles/step)",
                title=f"{lname} f_peak distribution (epoch {int(epoch)})",
            )
            save_hist_line(
                os.path.join(base, f"{lname}_hist_r0.png"),
                metrics["r0"].reshape(-1),
                xlabel="r0 (=A_norm(0))",
                title=f"{lname} r0 distribution (epoch {int(epoch)})",
            )
            save_hist_line(
                os.path.join(base, f"{lname}_hist_rpi.png"),
                metrics["rpi"].reshape(-1),
                xlabel="rpi (=A_norm(pi))",
                title=f"{lname} rpi distribution (epoch {int(epoch)})",
            )

            # selected neuron plots: only for hidden layers (analysis_neuron_indices excludes output)
            if analysis_neuron_indices is not None and li < len(analysis_neuron_indices):
                idx_list = analysis_neuron_indices[li]
                if len(idx_list) > 0:
                    sel_root = ensure_dir(os.path.join(base, f"{lname}_selected_neurons"))

                    alpha = metrics["alpha"]
                    gate = metrics["gate"]
                    w_mix = metrics["w_mix"]
                    s_vec = metrics["s"]
                    A = metrics["A"]
                    A_norm = metrics["A_norm"]

                    for nidx in idx_list:
                        nidx = int(nidx)
                        if nidx < 0 or nidx >= int(alpha.shape[0]):
                            continue
                        ndir = ensure_dir(os.path.join(sel_root, f"neuron_{nidx}"))

                        # per-neuron summary row
                        row = {
                            "r0": float(metrics["r0"][nidx]),
                            "rpi": float(metrics["rpi"][nidx]),
                            "f_peak": float(metrics["f_peak"][nidx]),
                            "bandwidth": float(metrics["bandwidth"][nidx]),
                            "class_id": int(metrics["class_id"][nidx]),
                            "class": str(_RDH_CLASS_NAMES[int(metrics["class_id"][nidx])]),
                            "s": float(s_vec[nidx]),
                            "D_int": int(metrics["D_int"][nidx]) if nidx < metrics["D_int"].shape[0] else None,
                        }
                        save_json(os.path.join(ndir, "meta.json"), row)

                        _save_rdh_transfer_selected_neuron_plots(
                            out_dir=ndir,
                            layer_name=lname,
                            neuron_idx=nidx,
                            freqs=freqs,
                            alpha_row=alpha[nidx],
                            gate_row=gate[nidx],
                            w_row=w_mix[nidx],
                            s_val=float(s_vec[nidx]),
                            A_row=A[nidx],
                            A_norm_row=A_norm[nidx],
                            summary_row=row,
                        )


def _save_rdh_transfer_trends(out_dir: str, transfer_history: Dict[str, List[Dict[str, Any]]]) -> None:
    """Save final-epoch trend + delta plots from per-epoch transfer summaries."""
    trend_dir = ensure_dir(os.path.join(out_dir, "transfer", "trend"))

    for lname, hist in transfer_history.items():
        if not hist:
            continue
        epochs = [int(h.get("epoch", i + 1)) for i, h in enumerate(hist)]

        # Fractions
        frac_lp = [float(h.get("fractions", {}).get("LP", 0.0)) for h in hist]
        frac_bp = [float(h.get("fractions", {}).get("BP", 0.0)) for h in hist]
        frac_hp = [float(h.get("fractions", {}).get("HP", 0.0)) for h in hist]
        frac_mx = [float(h.get("fractions", {}).get("mixed", 0.0)) for h in hist]

        save_line_plot(
            os.path.join(trend_dir, f"{lname}_ratio.png"),
            {"LP": frac_lp, "BP": frac_bp, "HP": frac_hp, "mixed": frac_mx},
            x=epochs,
            xlabel="epoch",
            ylabel="fraction",
            title=f"{lname} filter-type fraction by epoch",
        )

        def _delta(seq):
            out = [0.0]
            for i in range(1, len(seq)):
                out.append(float(seq[i]) - float(seq[i - 1]))
            return out

        save_line_plot(
            os.path.join(trend_dir, f"{lname}_ratio_delta.png"),
            {"LP": _delta(frac_lp), "BP": _delta(frac_bp), "HP": _delta(frac_hp), "mixed": _delta(frac_mx)},
            x=epochs,
            xlabel="epoch",
            ylabel="Δ fraction",
            title=f"{lname} Δ(filter-type fraction) by epoch",
        )

        # Means
        mean_f_peak = [float(h.get("means", {}).get("f_peak", 0.0)) for h in hist]
        mean_r0 = [float(h.get("means", {}).get("r0", 0.0)) for h in hist]
        mean_rpi = [float(h.get("means", {}).get("rpi", 0.0)) for h in hist]

        save_line_plot(
            os.path.join(trend_dir, f"{lname}_f_peak_mean.png"),
            {"f_peak": mean_f_peak},
            x=epochs,
            xlabel="epoch",
            ylabel="f_peak (cycles/step)",
            title=f"{lname} mean f_peak by epoch",
        )
        save_line_plot(
            os.path.join(trend_dir, f"{lname}_f_peak_mean_delta.png"),
            {"Δf_peak": _delta(mean_f_peak)},
            x=epochs,
            xlabel="epoch",
            ylabel="Δ f_peak",
            title=f"{lname} Δ(mean f_peak) by epoch",
        )

        save_line_plot(
            os.path.join(trend_dir, f"{lname}_r0_mean.png"),
            {"r0": mean_r0},
            x=epochs,
            xlabel="epoch",
            ylabel="r0",
            title=f"{lname} mean r0 by epoch",
        )
        save_line_plot(
            os.path.join(trend_dir, f"{lname}_r0_mean_delta.png"),
            {"Δr0": _delta(mean_r0)},
            x=epochs,
            xlabel="epoch",
            ylabel="Δ r0",
            title=f"{lname} Δ(mean r0) by epoch",
        )

        save_line_plot(
            os.path.join(trend_dir, f"{lname}_rpi_mean.png"),
            {"rpi": mean_rpi},
            x=epochs,
            xlabel="epoch",
            ylabel="rpi",
            title=f"{lname} mean rpi by epoch",
        )
        save_line_plot(
            os.path.join(trend_dir, f"{lname}_rpi_mean_delta.png"),
            {"Δrpi": _delta(mean_rpi)},
            x=epochs,
            xlabel="epoch",
            ylabel="Δ rpi",
            title=f"{lname} Δ(mean rpi) by epoch",
        )

        # CSV dump for convenience
        csv_path = os.path.join(trend_dir, f"{lname}_metrics.csv")
        import csv
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "epoch",
                "frac_LP",
                "frac_BP",
                "frac_HP",
                "frac_mixed",
                "mean_f_peak",
                "mean_r0",
                "mean_rpi",
                "mean_bandwidth",
                "mean_s",
                "mean_D_int",
            ])
            for h in hist:
                w.writerow([
                    int(h.get("epoch", 0)),
                    float(h.get("fractions", {}).get("LP", 0.0)),
                    float(h.get("fractions", {}).get("BP", 0.0)),
                    float(h.get("fractions", {}).get("HP", 0.0)),
                    float(h.get("fractions", {}).get("mixed", 0.0)),
                    float(h.get("means", {}).get("f_peak", 0.0)),
                    float(h.get("means", {}).get("r0", 0.0)),
                    float(h.get("means", {}).get("rpi", 0.0)),
                    float(h.get("means", {}).get("bandwidth", 0.0)),
                    float(h.get("means", {}).get("s", 0.0)),
                    float(h.get("means", {}).get("D_int", 0.0)),
                ])
def _build_params_json(model: MultiLayerSNNClassifier, layer_names: Sequence[str]) -> Dict[str, Any]:
    """Build params.json content.

    experiment.md requires **active-branch** parameter counts only.
    """

    layers: Dict[str, Any] = {}
    total_breakdown = {k: 0 for k in PARAM_CATEGORIES}
    for lname, layer in zip(layer_names, model.layers):
        bd = layer_active_param_breakdown(layer)
        for k in total_breakdown:
            total_breakdown[k] += int(bd.get(k, 0))
        layers[lname] = {
            "params": int(sum(int(v) for v in bd.values())),
            "breakdown": bd,
        }

    total_active = int(count_active_parameters(model))
    return {
        # For this file, "total_params" means total **active** parameters.
        "total_params": total_active,
        "layers": layers,
        "active_param_breakdown": total_breakdown,
    }


def run_freq_analysis(
    dataset: str,
    model: str,
    out_root: str,
    data_root: str,
    hidden: Sequence[int],
    epochs: int = 50,
    soft_mask_epochs: Optional[int] = None,
    stabilize_epochs: int = 0,
    ste_epochs: int = 0,
    batch_size: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    weight_decay_dend_soma: Optional[float] = None,
    seed: int = 0,
    S_min: float = 1.0,
    S_max: float = 8.0,
    th_len: int = 4,
    v_th: float = 1.0,
    v_pre: float = 1.0,
    cifar_mode: str = "parallel",
    T_event: int = 250,
    num_workers: int = 4,
    download: bool = False,
    plot_every: int = 5,
    analysis_every: int = 5,
    convergence_every: int = 5,
    analysis_neurons: Optional[Sequence[int]] = None,
    fft_band_edges: Optional[Sequence[float]] = None,
    fft_band_reduce: str = "mean",
    lambda_ortho: float = 0.0,
    lambda_s: float = 0.0,
    exp_name: Optional[str] = None,
    timestamp: Optional[str] = None,
    device: str = "auto",
) -> str:
    """Run the frequency analysis experiment as specified in src/freq_analysis/experiment.md."""

    # --------------------------
    # Validation / setup
    # --------------------------
    hidden = [int(h) for h in hidden]
    if len(hidden) < 1:
        raise ValueError("hidden must contain at least one hidden layer size")
    # Branch count (tensor shape) is derived ONLY from S_max (no separate dendritic arg).
    S_min = float(S_min)
    S_max = float(S_max)
    if not (S_max > 0.0):
        raise ValueError(f"S_max must be > 0 (got S_max={S_max})")
    if not (1.0 <= S_min <= S_max):
        raise ValueError(f"Require 1 <= S_min <= S_max (got S_min={S_min}, S_max={S_max})")

    branch = int(derive_branch_from_S_max(S_max))

    # analysis_neurons semantics (수정점):
    # - provided: treated as per-hidden-layer *sample counts*
    # - omitted and analysis is enabled: default 5 per hidden layer
    # We sample neuron indices once and record them into config.json.
    analysis_neurons_source: Optional[str] = None
    analysis_neurons_seed: Optional[int] = None
    analysis_neurons_counts: Optional[List[int]] = None
    analysis_neuron_indices: Optional[List[List[int]]] = None

    if int(analysis_every) > 0:
        if analysis_neurons is None:
            analysis_neurons_counts = [5 for _ in hidden]
            analysis_neurons_source = "default"
        else:
            analysis_neurons_counts = [int(n) for n in analysis_neurons]
            analysis_neurons_source = "cli"
            if len(analysis_neurons_counts) != len(hidden):
                raise ValueError(
                    f"analysis_neurons length must match number of hidden layers ({len(hidden)}), got {len(analysis_neurons_counts)}"
                )
        for li, (h, c) in enumerate(zip(hidden, analysis_neurons_counts)):
            if int(h) <= 0:
                raise ValueError(f"hidden[{li}] must be > 0, got {h}")
            if int(c) < 0:
                raise ValueError(f"analysis_neurons[{li}] must be >= 0 (got {c})")
            if int(c) > int(h):
                raise ValueError(
                    f"analysis_neurons[{li}] (count={c}) cannot exceed hidden[{li}]={h}"
                )

        analysis_neurons_seed = int(seed) + 30000
        rng = np.random.RandomState(analysis_neurons_seed)
        analysis_neuron_indices = []
        for h, c in zip(hidden, analysis_neurons_counts):
            if int(c) == 0:
                analysis_neuron_indices.append([])
            else:
                idx = rng.choice(int(h), size=int(c), replace=False)
                analysis_neuron_indices.append([int(i) for i in np.sort(idx)])

    if fft_band_reduce not in ("mean", "sum", "l2", "max"):
        raise ValueError(f"fft_band_reduce must be one of mean/sum/l2/max, got {fft_band_reduce}")

    dev = get_device(device)
    set_seed(int(seed))

    # --------------------------
    # IO paths
    # --------------------------
    ts = str(timestamp) if timestamp is not None else now_timestamp_seoul()
    builder_name = _model_name_to_builder(model)
    default_exp_name = f"freq_analysis-{dataset}-{model}"
    exp_name_final = (exp_name or default_exp_name).replace(" ", "").replace("/", "-")
    out_dir = ensure_dir(os.path.join(out_root, f"{exp_name_final}_{ts}"))

    # --------------------------
    # Data
    # --------------------------
    train_loader, test_loader, num_classes, input_dim, T, dt_ms = _load_dataset(
        dataset=dataset,
        data_root=data_root,
        batch_size=batch_size,
        num_workers=num_workers,
        download=download,
        cifar_mode=cifar_mode,
        T_event=T_event,
        seed=int(seed),
    )

    # --------------------------
    # FFT band binning precompute
    # --------------------------
    fft_band_ranges: Optional[List[Tuple[int, int]]] = None
    if fft_band_edges is not None:
        fft_band_edges = [float(x) for x in fft_band_edges]
        fft_band_ranges = band_edges_to_bin_ranges(T, fft_band_edges, d=1.0)

    freqs = rfft_freqs(T, d=1.0)

    # R-DH transfer analysis (dendrite_input -> soma_input)
    transfer_enabled = (builder_name == "my-r-dh-snn")
    transfer_thresholds: Dict[str, float] = {
        "lp_peak_max": 0.05,
        "hp_peak_min": 0.45,
        "lp_r0_min": 0.7,
        "lp_rpi_max": 0.5,
        "hp_r0_max": 0.3,
        "hp_rpi_min": 0.7,
        "bp_r0_max": 0.5,
        "bp_rpi_max": 0.5,
        "db3_thresh": float(2.0 ** (-0.5)),
    }

    # --------------------------
    # Probes (one per label)
    # --------------------------
    probes = _select_probe_samples(test_loader, num_classes=num_classes, seed=int(seed))
    if len(probes) == 0:
        raise RuntimeError("Failed to select probe samples (no labels found in test loader)")

    for lbl, info in probes.items():
        label_dir = ensure_dir(os.path.join(out_dir, f"label_{lbl}"))
        visualize_input_sequence(
            dataset=dataset,
            x_seq=info["x"],
            out_dir=label_dir,
            fft_band_edges=fft_band_edges,
            fft_band_reduce=fft_band_reduce,
        )

    # --------------------------
    # Model
    # --------------------------
    cfg = SNNConfig(
        model_name=builder_name,
        input_dim=input_dim,
        # SNNConfig.hidden_dim is not used by build_layer for the proposed models;
        # keep a sensible value for logging/debugging.
        hidden_dim=hidden[0] if len(hidden) > 0 else num_classes,
        num_classes=num_classes,
        branch=int(branch),
        S_min=float(S_min),
        S_max=float(S_max),
        th_len=int(th_len),
        v_th=float(v_th),
        v_reset=float(v_th),
        v_pre=float(v_pre),
    )

    dims = [input_dim] + hidden + [num_classes]
    layers: List[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(build_layer(builder_name, dims[i], dims[i + 1], cfg))

    # Output layer is used with membrane-potential readout (mean over time of soma_state).
    # Disable spiking/reset in the output layer to avoid clamping logits.
    _disable_output_spikes_(layers[-1])

    net = MultiLayerSNNClassifier(layers).to(dev)
    layer_names = _layer_names(hidden)

    # --------------------------
    # Config / params placeholders
    # --------------------------
    # Derived seeds must reflect the actual values used in data loaders / analysis helpers.
    derived_seeds: Dict[str, Optional[int]] = {
        "train_dataloader": int(seed),
        "test_dataloader": int(seed) + 1,
        "worker_init_base": int(seed),
        "probe_selection": int(seed),
        "analysis_neurons": int(analysis_neurons_seed) if analysis_neurons_seed is not None else None,
    }

    analysis_neurons_cfg: Optional[Dict[str, Any]] = None
    if analysis_neurons_counts is not None and analysis_neuron_indices is not None:
        analysis_neurons_cfg = {
            "source": analysis_neurons_source,
            "seed": int(analysis_neurons_seed) if analysis_neurons_seed is not None else None,
            "counts": list(analysis_neurons_counts),
            "indices": analysis_neuron_indices,
        }
    soft_e = int(epochs) if soft_mask_epochs is None else int(soft_mask_epochs)
    stb_e = int(stabilize_epochs)
    ste_e = int(ste_epochs)
    if ste_e < 0:
        raise ValueError(f"ste_epochs must be >= 0 (got ste_epochs={ste_e})")
    # STE only matters when we actually harden (i.e., stage B exists).
    if int(stb_e) <= 0:
        ste_e = 0
    if ste_e > int(soft_e):
        ste_e = int(soft_e)
    total_e = int(soft_e + stb_e)



    config: Dict[str, Any] = {
        "exp_name": exp_name_final,
        "timestamp": ts,
        "dataset": dataset,
        "model": model,
        "builder_name": builder_name,
        "readout": "mem",
        "hidden": hidden,
        "num_classes": num_classes,
        "input_dim": input_dim,
        "T": T,
        "dt_ms": dt_ms,
        "S_ms": float(T) * float(dt_ms),
        "dataset_options": {
            "cifar_mode": cifar_mode if dataset == "s-cifar10" else None,
            "T_event": int(T_event) if dataset in ("SHD", "SSC") else None,
        },
        "branch": int(branch),
        "S_min": float(S_min),
        "S_max": float(S_max),
        "th_len": int(th_len),
        "v_th": float(v_th),
        "v_pre": float(v_pre),
        "readout_is_proposed": True,
        "seed": int(seed),
        "derived_seeds": derived_seeds,
        "training": {
            "epochs": int(total_e),
            "soft_mask_epochs": int(soft_e),
            "stabilize_epochs": int(stb_e),
            "ste_epochs": int(ste_e),
            "batch_size": int(batch_size),
            "lr": float(lr),
            "weight_decay": float(weight_decay),
            "weight_decay_dend_soma": (None if weight_decay_dend_soma is None else float(weight_decay_dend_soma)),
            "lambda_ortho": float(lambda_ortho),
            "lambda_s": float(lambda_s),
        },
        "schedules": {
            "plot_every": int(plot_every),
            "analysis_every": int(analysis_every),
            "convergence_every": int(convergence_every),
        },
        "analysis_neurons": analysis_neurons_cfg,
        "fft": {
            "band_edges": list(fft_band_edges) if fft_band_edges is not None else None,
            "band_reduce": fft_band_reduce,
        },
        "probe": {
            str(lbl): {
                "dataset_index": int(info["dataset_index"]),
                "label": int(lbl),
            }
            for lbl, info in probes.items()
        },
        "signals": {
            "keys": list(SIGNALS),
            "mapping": _signal_mapping(builder_name),
        },
        "transfer_analysis": (
            {
                "enabled": True,
                "freq_axis_unit": "cycles/step",
                "thresholds": transfer_thresholds,
                "doc": "paper/proposed/r_dh_snn_transfer_analysis.md",
            }
            if transfer_enabled
            else {"enabled": False}
        ),
        "device": str(dev),
        "backend_flags": get_backend_flags(),
    }
    save_json(os.path.join(out_dir, "config.json"), config)

    # --------------------------
    # Optimizer / criterion
    # --------------------------
    from src.common.optim import build_adamw

    optimizer, opt_info = build_adamw(
        net,
        lr=float(lr),
        weight_decay=float(weight_decay),
        weight_decay_dend_soma=weight_decay_dend_soma,
    )
    criterion = nn.CrossEntropyLoss()

    # --------------------------
    # Convergence state containers
    # --------------------------
    # delta_hist_exact[label][layer_name][signal] = list of numpy arrays (index_dim,)
    delta_hist_exact: Dict[int, Dict[str, Dict[str, List[np.ndarray]]]] = {}
    delta_hist_binned: Optional[Dict[int, Dict[str, Dict[str, List[np.ndarray]]]]] = None
    if fft_band_ranges is not None:
        delta_hist_binned = {}

    # prev spectra stored on CPU to avoid large GPU memory usage
    prev_spec_exact: Dict[int, Dict[str, Dict[str, Optional[torch.Tensor]]]] = {}
    prev_spec_binned: Optional[Dict[int, Dict[str, Dict[str, Optional[torch.Tensor]]]]] = None
    if fft_band_ranges is not None:
        prev_spec_binned = {}

    for lbl in probes.keys():
        delta_hist_exact[lbl] = {}
        prev_spec_exact[lbl] = {}
        if delta_hist_binned is not None:
            delta_hist_binned[lbl] = {}
        if prev_spec_binned is not None:
            prev_spec_binned[lbl] = {}

        for lname in layer_names:
            delta_hist_exact[lbl][lname] = {s: [] for s in SIGNALS}
            prev_spec_exact[lbl][lname] = {s: None for s in SIGNALS}
            if delta_hist_binned is not None:
                delta_hist_binned[lbl][lname] = {s: [] for s in SIGNALS}
            if prev_spec_binned is not None:
                prev_spec_binned[lbl][lname] = {s: None for s in SIGNALS}

    # --------------------------
    # Mask history (active-only convergence + neuron plots)
    # --------------------------
    # mask_hist[layer_name] = list over epoch of flattened mask (N*Dmax,)
    mask_hist: Dict[str, List[np.ndarray]] = {lname: [] for lname in layer_names}

    # --------------------------
    # Selected-neuron parameter history (for final-epoch trajectory plots)
    # --------------------------
    # neuron_param_hist[layer_name][neuron_idx][param] = list over epoch
    neuron_param_hist: Dict[str, Dict[int, Dict[str, List[Any]]]] = {}
    if analysis_neuron_indices is not None:
        for k, idx_list in enumerate(analysis_neuron_indices):
            lname = layer_names[k]
            neuron_param_hist[lname] = {}
            for nidx in idx_list:
                neuron_param_hist[lname][int(nidx)] = {}


    # --------------------------
    # Selected-neuron weight history (for final-epoch per-epoch distribution overlays)
    # --------------------------
    # neuron_weight_hist[layer_name][neuron_idx] = list over epoch of raw weight snapshots
    # layer_w_in_hist[layer_name] = list over epoch of layer connection weights W_in (R_DH only)
    neuron_weight_hist: Dict[str, Dict[int, List[np.ndarray]]] = {}
    layer_w_in_hist: Dict[str, List[np.ndarray]] = {}
    if analysis_neuron_indices is not None:
        for k, idx_list in enumerate(analysis_neuron_indices):
            lname = layer_names[k]
            neuron_weight_hist[lname] = {int(nidx): [] for nidx in idx_list}
            layer = net.layers[k]
            if hasattr(layer, "W_in"):
                layer_w_in_hist[lname] = []


    # --------------------------
    # R-DH transfer analysis history (for trend plots)
    # --------------------------
    transfer_history: Dict[str, List[Dict[str, Any]]] = {}

    # --------------------------
    # Training loop
    # --------------------------
    metrics_path = os.path.join(out_dir, "metrics.csv")
    with open(metrics_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "test_loss", "test_acc", "sec"])

        best_test = -1.0
        best_epoch = 0
        history_epochs: List[int] = []
        history_train_acc: List[float] = []
        history_test_acc: List[float] = []
        history_train_loss: List[float] = []
        history_test_loss: List[float] = []

        t0 = time.time()
        pbar = tqdm(range(1, int(total_e) + 1), desc=f"{dataset}:{model}", total=int(total_e), leave=True)
        for epoch in pbar:
            # STE schedule (last `ste_e` epochs of stage A): forward hard / backward soft.
            if int(stb_e) > 0:
                from src.common.model_utils import set_ste_mode_
                ste_on = int(ste_e) > 0 and (int(epoch) >= int(soft_e) - int(ste_e) + 1) and (int(epoch) <= int(soft_e))
                set_ste_mode_(net, bool(ste_on))
            else:
                # No hardening stage -> keep STE off.
                from src.common.model_utils import set_ste_mode_
                set_ste_mode_(net, False)


            if int(stb_e) > 0 and int(epoch) == int(soft_e) + 1:
                from src.common.model_utils import harden_variable_branches_

                harden_variable_branches_(net)
            epoch_start = time.time()

            tr_loss = _train_one_epoch(
                net,
                train_loader,
                optimizer,
                criterion,
                device=dev,
                lambda_ortho=float(lambda_ortho),
                lambda_s=float(lambda_s),
            )

            # Full eval (train/test) for clean curves.
            train_loss, train_acc = _evaluate_classifier(net, train_loader, criterion, dev)
            test_loss, test_acc = _evaluate_classifier(net, test_loader, criterion, dev)

            history_epochs.append(epoch)
            history_train_loss.append(float(train_loss))
            history_test_loss.append(float(test_loss))
            history_train_acc.append(float(train_acc))
            history_test_acc.append(float(test_acc))

            if test_acc > best_test:
                best_test = float(test_acc)
                best_epoch = int(epoch)

            # Distributions
            if int(plot_every) > 0 and (epoch % int(plot_every) == 0):
                _save_distribution_plots(out_dir, net, epoch, layer_names=layer_names, fft_model_tag=model)

            # --------------------------------------------------
            # Record masks + selected-neuron parameter snapshots (every epoch)
            # --------------------------------------------------
            for lname, layer in zip(layer_names, net.layers):
                if hasattr(layer, "soft_mask") and callable(getattr(layer, "soft_mask")):
                    try:
                        m = layer.soft_mask(torch.float32).detach().cpu().numpy().reshape(-1)  # type: ignore
                    except Exception:
                        m = np.zeros((0,), dtype=np.float32)
                else:
                    m = np.zeros((0,), dtype=np.float32)
                mask_hist[lname].append(np.asarray(m, dtype=np.float32))

            if analysis_neuron_indices is not None and neuron_param_hist:
                for k, idx_list in enumerate(analysis_neuron_indices):
                    lname = layer_names[k]
                    layer = net.layers[k]
                    # R_DH: record axon-shared layer connection weights once per epoch (same for all neurons in the layer)
                    try:
                        if hasattr(layer, "W_in"):
                            w_in = getattr(layer, "W_in").detach().cpu().to(torch.float32).numpy().reshape(-1)  # type: ignore
                            layer_w_in_hist.setdefault(lname, []).append(np.asarray(w_in, dtype=float))
                    except Exception:
                        pass

                    for neuron_idx in idx_list:
                        nidx = int(neuron_idx)
                        slot = neuron_param_hist[lname][nidx]

                        # structure: s, D_int
                        if hasattr(layer, "s") and callable(getattr(layer, "s")):
                            try:
                                s_vec = layer.s().detach().cpu().to(torch.float32).numpy().reshape(-1)  # type: ignore
                                if 0 <= nidx < int(s_vec.shape[0]):
                                    slot.setdefault("s", []).append(float(s_vec[nidx]))
                            except Exception:
                                pass
                        if hasattr(layer, "d_int") and callable(getattr(layer, "d_int")):
                            try:
                                d_vec = layer.d_int().detach().cpu().to(torch.int64).numpy().reshape(-1)  # type: ignore
                                if 0 <= nidx < int(d_vec.shape[0]):
                                    slot.setdefault("D_int", []).append(int(d_vec[nidx]))
                            except Exception:
                                pass

                        # timing / resonance
                        if hasattr(layer, "alpha_branch") and callable(getattr(layer, "alpha_branch")):
                            try:
                                a_mat = layer.alpha_branch().detach().cpu().to(torch.float32).numpy()  # type: ignore
                                if a_mat.ndim == 2 and 0 <= nidx < int(a_mat.shape[0]):
                                    slot.setdefault("alpha", []).append(np.asarray(a_mat[nidx], dtype=float).reshape(-1))
                            except Exception:
                                pass
                        if hasattr(layer, "beta_soma") and callable(getattr(layer, "beta_soma")):
                            try:
                                b_vec = layer.beta_soma().detach().cpu().to(torch.float32).numpy().reshape(-1)  # type: ignore
                                if 0 <= nidx < int(b_vec.shape[0]):
                                    slot.setdefault("beta", []).append(float(b_vec[nidx]))
                            except Exception:
                                pass
                        if hasattr(layer, "tau") and callable(getattr(layer, "tau")):
                            try:
                                t_mat = layer.tau().detach().cpu().to(torch.float32).numpy()  # type: ignore
                                if t_mat.ndim == 2 and 0 <= nidx < int(t_mat.shape[0]):
                                    slot.setdefault("tau", []).append(np.asarray(t_mat[nidx], dtype=float).reshape(-1))
                            except Exception:
                                pass
                        if hasattr(layer, "omega") and callable(getattr(layer, "omega")):
                            try:
                                o_mat = layer.omega().detach().cpu().to(torch.float32).numpy()  # type: ignore
                                if o_mat.ndim == 2 and 0 <= nidx < int(o_mat.shape[0]):
                                    slot.setdefault("omega", []).append(np.asarray(o_mat[nidx], dtype=float).reshape(-1))
                            except Exception:
                                pass

                        # weights: store raw snapshots for final-epoch weight distribution overlays

                        try:
                            if hasattr(layer, "W_mix"):
                                Wm = getattr(layer, "W_mix")  # type: ignore
                                if 0 <= nidx < int(Wm.shape[0]):
                                    w_row = Wm[nidx].detach().cpu().to(torch.float32).numpy().reshape(-1)
                                    neuron_weight_hist[lname][nidx].append(np.asarray(w_row, dtype=float))
                            elif hasattr(layer, "W") and hasattr(layer, "branch") and hasattr(layer, "input_dim"):
                                W = getattr(layer, "W")  # type: ignore
                                Dm = int(getattr(layer, "branch"))
                                in_dim = int(getattr(layer, "input_dim"))
                                row0 = int(nidx) * int(Dm)
                                rows = W.view(-1, in_dim)[row0 : row0 + int(Dm), :]
                                neuron_weight_hist[lname][nidx].append(rows.detach().cpu().to(torch.float32).numpy())
                            elif hasattr(layer, "fc") and isinstance(getattr(layer, "fc"), nn.Linear):
                                fc: nn.Linear = getattr(layer, "fc")
                                if 0 <= nidx < int(fc.weight.shape[0]):
                                    neuron_weight_hist[lname][nidx].append(
                                        fc.weight[nidx].detach().cpu().to(torch.float32).numpy().reshape(-1)
                                    )
                        except Exception:
                            pass

            # --------------------------------------------------
            # R-DH transfer analysis (every epoch; plots on plot_every)
            # --------------------------------------------------
            if transfer_enabled:
                _save_rdh_transfer_epoch(
                    out_dir=out_dir,
                    model=net,
                    epoch=int(epoch),
                    layer_names=layer_names,
                    freqs=freqs,
                    analysis_neuron_indices=analysis_neuron_indices,
                    plot_every=int(plot_every),
                    transfer_history=transfer_history,
                    thresholds=transfer_thresholds,
                )



            # --------------------------------------------------
            # Per-label probe forward + Δ update (every epoch)
            # --------------------------------------------------
            net.eval()
            for lbl, info in probes.items():
                x_seq = info["x"].to(dev).unsqueeze(0)  # (1,T,C)
                _, recs = net.forward_with_records(x_seq)

                # Update Δ history for each layer/signal
                for lname, rec in zip(layer_names, recs):
                    for sig in SIGNALS:
                        x_time = _prep_time_matrix(rec[sig]).to(torch.float32)
                        S = rfft_log_mag(x_time, dim=-1)  # (index_dim,F)

                        prev = prev_spec_exact[lbl][lname][sig]
                        if prev is None:
                            delta = torch.zeros(S.shape[0], device=S.device, dtype=torch.float32)
                        else:
                            prev_gpu = prev.to(device=S.device, dtype=S.dtype)
                            delta = torch.linalg.vector_norm(S - prev_gpu, ord=2, dim=-1)
                        delta_hist_exact[lbl][lname][sig].append(delta.detach().cpu().to(torch.float32).numpy())
                        # store prev (CPU, float16 to reduce RAM)
                        prev_spec_exact[lbl][lname][sig] = S.detach().cpu().to(torch.float16)

                        if fft_band_ranges is not None and delta_hist_binned is not None and prev_spec_binned is not None:
                            Sb = bin_spectrum(S, fft_band_ranges, dim=-1, reduce=fft_band_reduce)
                            prevb = prev_spec_binned[lbl][lname][sig]
                            if prevb is None:
                                delt_b = torch.zeros(Sb.shape[0], device=Sb.device, dtype=torch.float32)
                            else:
                                prevb_gpu = prevb.to(device=Sb.device, dtype=Sb.dtype)
                                delt_b = torch.linalg.vector_norm(Sb - prevb_gpu, ord=2, dim=-1)
                            delta_hist_binned[lbl][lname][sig].append(delt_b.detach().cpu().to(torch.float32).numpy())
                            prev_spec_binned[lbl][lname][sig] = Sb.detach().cpu().to(torch.float16)

                # Analysis (selected neurons + layer outputs)
                if int(analysis_every) > 0 and (epoch % int(analysis_every) == 0):
                    label_dir = os.path.join(out_dir, f"label_{lbl}")
                    epoch_dir = _ensure_epoch_dir(label_dir, epoch)

                    # 6.5: per-hidden-layer selected neuron analysis
                    if analysis_neuron_indices is not None:
                        for k, idx_list in enumerate(analysis_neuron_indices):
                            lname = layer_names[k]
                            # Active-only handling for dendrite signals uses the per-epoch soft mask history.
                            mask_for_layer = mask_hist.get(lname)
                            rec = recs[k]
                            for neuron_idx in idx_list:
                                neuron_dir = os.path.join(epoch_dir, f"{lname}_neuron_{int(neuron_idx)}")
                                _save_neuron_analysis(
                                    neuron_dir=neuron_dir,
                                    epoch=epoch,
                                    layer_name=lname,
                                    neuron_idx=int(neuron_idx),
                                    layer=net.layers[k],
                                    rec=rec,
                                    delta_hist_exact=delta_hist_exact[lbl][lname],
                                    delta_hist_binned=(
                                        delta_hist_binned[lbl][lname] if delta_hist_binned is not None else None
                                    ),
                                    Dmax=int(branch),
                                    freqs=freqs,
                                    fft_band_edges=fft_band_edges,
                                    fft_band_ranges=fft_band_ranges,
                                    fft_band_reduce=fft_band_reduce,
                                    mask_hist_layer=mask_for_layer,
                                )

                    # 6.6: layer output heatmaps for ALL layers (hidden + output)
                    out_heat_dir = ensure_dir(os.path.join(epoch_dir, "layer_output"))
                    for lname, rec in zip(layer_names, recs):
                        out_sig = rec["output"][0]  # (T,N)
                        _save_layer_output_heatmaps(
                            out_dir=out_heat_dir,
                            layer_name=lname,
                            out_spk=out_sig,
                            freqs=freqs,
                            fft_band_edges=fft_band_edges,
                            fft_band_ranges=fft_band_ranges,
                            fft_band_reduce=fft_band_reduce,
                        )

            # Convergence plots are generated once at the final epoch (see below).

            # epoch row
            sec = float(time.time() - epoch_start)
            writer.writerow([epoch, train_loss, train_acc, test_loss, test_acc, sec])
            f.flush()

            pbar.set_postfix(
                train_loss=f"{float(train_loss):.4f}",
                train_acc=f"{float(train_acc):.4f}",
                test_loss=f"{float(test_loss):.4f}",
                test_acc=f"{float(test_acc):.4f}",
                sec=f"{sec:.1f}",
            )

        total_sec = float(time.time() - t0)

    # --------------------------------------------------
    # R-DH transfer analysis trend plots (final epoch only)
    # --------------------------------------------------
    if transfer_enabled and transfer_history:
        _save_rdh_transfer_trends(out_dir, transfer_history)


    # --------------------------------------------------
    # Final-epoch-only artifacts
    #   - convergence/ (Δ history across all epochs)
    #   - per-neuron distribution/ (parameter trajectories)
    # --------------------------------------------------
    final_epoch = int(total_e)
    epoch_x_full = np.arange(1, final_epoch + 1)

    # 6.5 distribution/: per-neuron parameter trajectories (final epoch only)
    if analysis_neuron_indices is not None and neuron_param_hist:
        for lbl in probes.keys():
            label_dir = os.path.join(out_dir, f"label_{lbl}")
            epoch_dir = _ensure_epoch_dir(label_dir, final_epoch)
            for k, idx_list in enumerate(analysis_neuron_indices):
                lname = layer_names[k]
                for nidx in idx_list:
                    neuron_dir = ensure_dir(os.path.join(epoch_dir, f"{lname}_neuron_{int(nidx)}"))
                    dist_dir = ensure_dir(os.path.join(neuron_dir, "distribution"))
                    _save_neuron_param_trajectories(
                        dist_dir=dist_dir,
                        layer_name=lname,
                        neuron_idx=int(nidx),
                        epoch_x=epoch_x_full,
                        param_hist=neuron_param_hist.get(lname, {}).get(int(nidx), {}),
                        Dmax=int(branch),
                        mask_hist_layer=mask_hist.get(lname),
                    )
                    # weights/: per-epoch weight distribution overlays (final epoch only)
                    weight_dir = ensure_dir(os.path.join(neuron_dir, "weight"))
                    w_hist = neuron_weight_hist.get(lname, {}).get(int(nidx), [])
                    if len(w_hist) > 0:
                        mask_for_w = mask_hist.get(lname) if (lname in layer_w_in_hist or (len(w_hist) > 0 and np.asarray(w_hist[0]).ndim == 2)) else None
                        active_w = _active_weights_over_epochs(
                            weight_hist=w_hist[: len(epoch_x_full)],
                            neuron_idx=int(nidx),
                            Dmax=int(branch),
                            mask_hist_layer=mask_for_w,
                        )

                        # R_DH: split plots (layer connection weights vs internal W_mix)
                        if lname in layer_w_in_hist and len(layer_w_in_hist.get(lname, [])) > 0:
                            _save_hist_overlay_lines(
                                out_path=os.path.join(weight_dir, "layer_weight.png"),
                                values_by_epoch=[np.asarray(v, dtype=float).reshape(-1) for v in layer_w_in_hist[lname][: len(epoch_x_full)]],
                                epoch_x=epoch_x_full,
                                bins=60,
                                title=f"{lname} layer connection weight (W_in) distribution by epoch",
                                xlabel="W_in",
                                density=True,
                            )
                            _save_hist_overlay_lines(
                                out_path=os.path.join(weight_dir, "mix_weight.png"),
                                values_by_epoch=active_w,
                                epoch_x=epoch_x_full,
                                bins=60,
                                title=f"{lname} neuron{int(nidx)} internal weight (W_mix, active) distribution by epoch",
                                xlabel="W_mix (active)",
                                density=True,
                            )
                        else:
                            # Other models: single dendrite weight plot
                            _save_hist_overlay_lines(
                                out_path=os.path.join(weight_dir, "weight.png"),
                                values_by_epoch=active_w,
                                epoch_x=epoch_x_full,
                                bins=60,
                                title=f"{lname} neuron{int(nidx)} weight distribution by epoch",
                                xlabel="weight",
                                density=True,
                            )

    # 6.4 convergence/: Δ plots (final epoch only)
    if int(convergence_every) > 0:
        for lbl in probes.keys():
            label_dir = os.path.join(out_dir, f"label_{lbl}")
            epoch_dir = _ensure_epoch_dir(label_dir, final_epoch)
            conv_base = ensure_dir(os.path.join(epoch_dir, "convergence"))

            # exact
            for lname in layer_names:
                layer_base = ensure_dir(os.path.join(conv_base, lname, "exact"))
                # Precompute (index_dim, epoch) mask matrix for dendrite signals if available.
                M: Optional[np.ndarray] = None
                if lname in mask_hist and len(mask_hist[lname]) == final_epoch:
                    try:
                        M = np.stack(mask_hist[lname], axis=1)
                    except Exception:
                        M = None

                for sig in SIGNALS:
                    hist_list = delta_hist_exact[lbl][lname][sig]
                    mat = np.stack(hist_list, axis=1)  # (index_dim, epoch)

                    if sig.startswith("dendrite_") and M is not None and M.shape == mat.shape:
                        denom = M.sum(axis=0) + 1e-12
                        mean = (M * mat).sum(axis=0) / denom
                        mat_plot = mat.astype(float, copy=True)
                        mat_plot[M <= 0.0] = np.nan
                    else:
                        mean = mat.mean(axis=0)
                        mat_plot = mat

                    save_line_plot(
                        os.path.join(layer_base, f"{sig}_mean_delta.png"),
                        {"mean": mean},
                        x=epoch_x_full,
                        xlabel="epoch",
                        ylabel="Δ",
                        title=f"label{lbl} {lname} {sig} mean Δ (exact)",
                    )
                    save_heatmap(
                        os.path.join(layer_base, f"{sig}_heatmap_delta.png"),
                        mat_plot,
                        xlabel="epoch",
                        ylabel="neuron (or neuron×branch)",
                        title=f"label{lbl} {lname} {sig} Δ heatmap (exact)",
                        use_log1p=True,
                    )

            # binned
            if delta_hist_binned is not None:
                for lname in layer_names:
                    layer_base = ensure_dir(os.path.join(conv_base, lname, "binned"))
                    M: Optional[np.ndarray] = None
                    if lname in mask_hist and len(mask_hist[lname]) == final_epoch:
                        try:
                            M = np.stack(mask_hist[lname], axis=1)
                        except Exception:
                            M = None
                    for sig in SIGNALS:
                        hist_list = delta_hist_binned[lbl][lname][sig]
                        mat = np.stack(hist_list, axis=1)

                        if sig.startswith("dendrite_") and M is not None and M.shape == mat.shape:
                            denom = M.sum(axis=0) + 1e-12
                            mean = (M * mat).sum(axis=0) / denom
                            mat_plot = mat.astype(float, copy=True)
                            mat_plot[M <= 0.0] = np.nan
                        else:
                            mean = mat.mean(axis=0)
                            mat_plot = mat

                        save_line_plot(
                            os.path.join(layer_base, f"{sig}_mean_delta.png"),
                            {"mean": mean},
                            x=epoch_x_full,
                            xlabel="epoch",
                            ylabel="Δ",
                            title=f"label{lbl} {lname} {sig} mean Δ (binned)",
                        )
                        save_heatmap(
                            os.path.join(layer_base, f"{sig}_heatmap_delta.png"),
                            mat_plot,
                            xlabel="epoch",
                            ylabel="neuron (or neuron×branch)",
                            title=f"label{lbl} {lname} {sig} Δ heatmap (binned)",
                            use_log1p=True,
                        )

    # --------------------------
    # Curves + summary + params
    # --------------------------
    save_line_plot(
        os.path.join(out_dir, "acc_curve.png"),
        {"train": history_train_acc, "test": history_test_acc},
        x=history_epochs,
        xlabel="epoch",
        ylabel="accuracy",
        title="Accuracy curve",
    )
    save_line_plot(
        os.path.join(out_dir, "loss_curve.png"),
        {"train": history_train_loss, "test": history_test_loss},
        x=history_epochs,
        xlabel="epoch",
        ylabel="loss",
        title="Loss curve",
    )

    summary = {
        # Metrics
        "best_test_acc": float(best_test),
        "best_epoch": int(best_epoch),
        "final_epoch": int(history_epochs[-1]) if history_epochs else None,
        "final_train_loss": float(history_train_loss[-1]) if history_train_loss else None,
        "final_test_loss": float(history_test_loss[-1]) if history_test_loss else None,
        "final_train_acc": float(history_train_acc[-1]) if history_train_acc else None,
        "final_test_acc": float(history_test_acc[-1]) if history_test_acc else None,
        # Major hyperparams / identifiers
        "dataset": dataset,
        "exp_name": exp_name_final,
        "timestamp": ts,
        "model": model,
        "hidden": hidden,
        "seed": int(seed),
        "training": {
            "epochs": int(total_e),
            "soft_mask_epochs": int(soft_e),
            "stabilize_epochs": int(stb_e),
            "ste_epochs": int(ste_e),
            "batch_size": int(batch_size),
            "lr": float(lr),
            "weight_decay": float(weight_decay),
            "weight_decay_dend_soma": (None if weight_decay_dend_soma is None else float(weight_decay_dend_soma)),
            "lambda_ortho": float(lambda_ortho),
            "lambda_s": float(lambda_s),
        },
        "dendrites": {
            "branch": int(branch),
            "S_min": float(S_min),
            "S_max": float(S_max),
            "th_len": int(th_len),
            "v_th": float(v_th),
            "v_pre": float(v_pre),
        },
        "schedules": {
            "plot_every": int(plot_every),
            "analysis_every": int(analysis_every),
            "convergence_every": int(convergence_every),
        },
        "fft": {
            "band_edges": list(fft_band_edges) if fft_band_edges is not None else None,
            "band_reduce": str(fft_band_reduce),
        },
        "analysis_neurons": analysis_neurons_cfg,
        # Runtime
        "device": str(dev),
        "total_time_sec": float(total_sec),
    }
    save_json(os.path.join(out_dir, "summary.json"), summary)

    params = _build_params_json(net, layer_names=layer_names)
    save_json(os.path.join(out_dir, "params.json"), params)

    # checkpoint
    torch.save(
        {
            "model": net.state_dict(),
            "config": config,
            "summary": summary,
        },
        os.path.join(out_dir, "final.pt"),
    )

    return out_dir
