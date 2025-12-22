from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split


DEFAULT_SSC_DATA_ROOT = Path(__file__).resolve().parents[2] / "shd_ssc_data" / "ssc"


class SSCH5Dataset(Dataset):
    def __init__(self, root: str | Path, split: str):
        self.root = Path(root)
        self.split = split
        self.file_path = self._resolve_path()
        self._length = self._resolve_length()

    def _resolve_path(self) -> Path:
        candidates = [
            self.root / f"SSC_{self.split}.h5",
            self.root / f"{self.split}.h5",
        ]
        for path in candidates:
            if path.exists():
                return path
        raise FileNotFoundError(f"No SSC h5 file found for split {self.split} in {self.root}")

    def _resolve_length(self) -> int:
        with h5py.File(self.file_path, "r") as f:
            if "labels" in f:
                return len(f["labels"])
            if "y" in f:
                return len(f["y"])
            if "spikes" in f and isinstance(f["spikes"], h5py.Group) and "times" in f["spikes"]:
                return len(f["spikes"]["times"])
            raise KeyError("Unable to infer dataset length from h5 structure")

    def __len__(self) -> int:
        return self._length

    def _read_label(self, file: h5py.File, idx: int) -> int:
        if "labels" in file:
            return int(file["labels"][idx])
        if "y" in file:
            return int(file["y"][idx])
        raise KeyError("labels/y dataset not found")

    def _read_events(self, file: h5py.File, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        if "spikes" in file and isinstance(file["spikes"], h5py.Group):
            group = file["spikes"]
            if "times" in group and "units" in group:
                return np.asarray(group["times"][idx]), np.asarray(group["units"][idx])
        if "times" in file and "units" in file:
            return np.asarray(file["times"][idx]), np.asarray(file["units"][idx])
        if "spikes" in file:
            spikes = np.asarray(file["spikes"][idx])
            if spikes.ndim == 2 and spikes.shape[1] == 2:
                return spikes[:, 0], spikes[:, 1]
        raise KeyError("No recognizable spikes/times/units datasets found")

    def __getitem__(self, idx: int):
        with h5py.File(self.file_path, "r") as f:
            times, units = self._read_events(f, idx)
            label = self._read_label(f, idx)
        return times.astype(np.float32), units.astype(np.int64), label


def events_to_dense(
    times: np.ndarray,
    units: np.ndarray,
    T: int,
    nb_units: int,
    max_time: float,
    dt: float | None,
    unit_mode: str = "as_is",
) -> torch.Tensor:
    if unit_mode == "reverse_1based":
        units = nb_units - units
    elif unit_mode == "as_is":
        pass
    else:
        raise ValueError(f"Unknown unit_mode: {unit_mode}")

    if dt is None:
        dt = max_time / float(T)
    bins = np.floor(times / dt).astype(np.int64)
    mask = (bins >= 0) & (bins < T) & (units >= 0) & (units < nb_units)
    bins = bins[mask]
    units = units[mask]
    dense = torch.zeros((T, nb_units), dtype=torch.float32)
    if bins.size > 0:
        idx = torch.tensor(np.stack([bins, units], axis=1), dtype=torch.int64)
        dense.index_put_(tuple(idx.t()), torch.ones(idx.shape[0]), accumulate=True)
    return dense


def _collate_fn(
    batch: Iterable[Tuple[np.ndarray, np.ndarray, int]],
    T: int,
    nb_units: int,
    max_time: float,
    dt: float | None,
    unit_mode: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    dense_list: List[torch.Tensor] = []
    labels: List[int] = []
    for times, units, label in batch:
        dense = events_to_dense(times, units, T=T, nb_units=nb_units, max_time=max_time, dt=dt, unit_mode=unit_mode)
        dense_list.append(dense)
        labels.append(label)
    x = torch.stack(dense_list, dim=0)
    y = torch.tensor(labels, dtype=torch.long)
    return x, y


def make_dataloaders(args):
    train_dataset = SSCH5Dataset(args.data_root, "train")
    val_path = Path(args.data_root) / "SSC_val.h5"
    test_dataset = SSCH5Dataset(args.data_root, "test")

    if val_path.exists():
        val_dataset = SSCH5Dataset(args.data_root, "val")
    else:
        val_len = max(1, int(len(train_dataset) * 0.1))
        train_len = len(train_dataset) - val_len
        train_dataset, val_dataset = random_split(
            train_dataset,
            [train_len, val_len],
            generator=torch.Generator().manual_seed(args.seed),
        )

    protocol_defaults = {
        "tc": {"T": 250, "max_time": 1.4, "dt": None, "unit_mode": "as_is"},
        "dh": {"T": args.T, "max_time": args.max_time, "dt": args.dt, "unit_mode": "reverse_1based"},
    }
    proto = protocol_defaults.get(args.protocol, protocol_defaults["tc"])
    T = args.T if args.T is not None else proto["T"]
    max_time = args.max_time if args.max_time is not None else proto["max_time"]
    dt = args.dt if args.dt is not None else proto["dt"]
    unit_mode = proto.get("unit_mode", "as_is")
    nb_units = args.nb_units

    collate = lambda batch: _collate_fn(batch, T=T, nb_units=nb_units, max_time=max_time, dt=dt, unit_mode=unit_mode)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        collate_fn=collate,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        collate_fn=collate,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        collate_fn=collate,
    )
    return train_loader, val_loader, test_loader
