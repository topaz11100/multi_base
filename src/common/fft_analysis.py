from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch


def rfft_log_mag(x: Union[np.ndarray, torch.Tensor], dim: int = -1) -> Union[np.ndarray, torch.Tensor]:
    """
    S = log(1 + |rFFT(x)|)

    Works for numpy (via numpy.fft.rfft) or torch (via torch.fft.rfft).
    """
    if isinstance(x, np.ndarray):
        X = np.fft.rfft(x, axis=dim)
        S = np.log1p(np.abs(X))
        return S
    X = torch.fft.rfft(x, dim=dim)
    S = torch.log1p(torch.abs(X))
    return S


def parse_band_edges(text: str) -> List[Tuple[int, int]]:
    """
    Parse band edges string like "0:4,5:8,9:16".
    Returns list of (lo, hi) inclusive.
    """
    edges: List[Tuple[int, int]] = []
    text = text.strip()
    if not text:
        return edges
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"Invalid band edge part: {part!r}")
        lo_s, hi_s = part.split(":")
        lo = int(lo_s)
        hi = int(hi_s)
        if hi < lo:
            raise ValueError(f"Invalid band edge range: {part!r}")
        edges.append((lo, hi))
    return edges


def rfft_freqs(T: int, d: float = 1.0) -> np.ndarray:
    """Return normalized rFFT frequency bins (cycles/step) for a 1D signal of length T."""
    T = int(T)
    if T <= 0:
        raise ValueError(f"T must be >= 1, got {T}")
    return np.fft.rfftfreq(T, d=float(d))


def band_edges_to_bin_ranges(
    T: int,
    band_edges: Sequence[float],
    d: float = 1.0,
) -> List[Tuple[int, int]]:
    """Convert band edges in cycles/step to inclusive (lo, hi) index ranges on rFFT bins.

    band_edges: [e0, e1, ..., eB] with e0 < e1 < ... < eB.
    Each band b corresponds to frequencies in [e_b, e_{b+1}) except the last band which is [e_{B-1}, e_B].
    """
    edges = [float(e) for e in band_edges]
    if len(edges) < 2:
        raise ValueError("band_edges must have at least 2 values")
    if any(np.isnan(edges)):
        raise ValueError("band_edges contains NaN")
    for i in range(len(edges) - 1):
        if not (edges[i] < edges[i + 1]):
            raise ValueError(f"band_edges must be strictly increasing, got {edges}")

    freqs = rfft_freqs(T, d=d)
    # NOTE:
    # - rfftfreq(T) has a maximum < 0.5 when T is odd.
    # - experiment.md defines band edges in normalized frequency (cycles/step) with an upper bound of 0.5.
    #   Therefore we validate against the theoretical Nyquist (0.5 / d), not freqs[-1].
    f_min_allowed = 0.0
    f_max_allowed = 0.5 / float(d)
    if edges[0] < f_min_allowed - 1e-12 or edges[-1] > f_max_allowed + 1e-12:
        raise ValueError(
            f"band_edges out of range for T={T}: expected within [{f_min_allowed:.6f}, {f_max_allowed:.6f}], got {edges[0]}..{edges[-1]}"
        )

    ranges: List[Tuple[int, int]] = []
    for b in range(len(edges) - 1):
        lo_e = edges[b]
        hi_e = edges[b + 1]
        if b == len(edges) - 2:
            mask = (freqs >= lo_e) & (freqs <= hi_e)
        else:
            mask = (freqs >= lo_e) & (freqs < hi_e)
        idx = np.nonzero(mask)[0]
        if idx.size == 0:
            raise ValueError(
                f"Empty FFT band for edge interval [{lo_e},{hi_e}) with T={T}. "
                "Choose edges that include at least one rFFT bin per band."
            )
        ranges.append((int(idx[0]), int(idx[-1])))
    return ranges


def bin_spectrum(
    S: Union[np.ndarray, torch.Tensor],
    band_edges: Sequence[Tuple[int, int]],
    dim: int = -1,
    reduce: str = "mean",
) -> Union[np.ndarray, torch.Tensor]:
    """
    Aggregate spectrum S over bands along dimension dim.

    S shape: (..., F)
    output: (..., B)
    """
    if len(band_edges) == 0:
        raise ValueError("band_edges is empty")

    if isinstance(S, np.ndarray):
        outs = []
        for lo, hi in band_edges:
            sl = [slice(None)] * S.ndim
            sl[dim] = slice(lo, hi + 1)
            band = S[tuple(sl)]
            if reduce == "mean":
                outs.append(band.mean(axis=dim))
            elif reduce == "sum":
                outs.append(band.sum(axis=dim))
            elif reduce == "l2":
                outs.append(np.sqrt((band * band).sum(axis=dim)))
            elif reduce == "max":
                outs.append(band.max(axis=dim))
            else:
                raise ValueError(f"Unknown reduce: {reduce}")
        return np.stack(outs, axis=dim)

    outs_t = []
    for lo, hi in band_edges:
        band = S.index_select(dim, torch.arange(lo, hi + 1, device=S.device))  # type: ignore
        if reduce == "mean":
            outs_t.append(band.mean(dim=dim))
        elif reduce == "sum":
            outs_t.append(band.sum(dim=dim))
        elif reduce == "l2":
            outs_t.append(torch.sqrt((band * band).sum(dim=dim)))
        elif reduce == "max":
            outs_t.append(band.max(dim=dim).values)
        else:
            raise ValueError(f"Unknown reduce: {reduce}")
    return torch.stack(outs_t, dim=dim)  # type: ignore
