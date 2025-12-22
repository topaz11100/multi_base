from pathlib import Path
from typing import Tuple

import numpy as np
import torch


DEFAULT_DATA_ROOT = Path(__file__).resolve().parents[2] / "shd_ssc_data" / "shd"


def _load_shd_raw(data_root: Path) -> Tuple[dict, dict]:
    try:
        import tonic
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("tonic is required for SHD dataset handling") from exc

    # tonic 캐시 문제 등으로 인한 경고가 뜰 수 있으나, 데이터 로딩 자체는 진행되도록 함
    data_root.mkdir(parents=True, exist_ok=True)
    train_ds = tonic.datasets.SHD(save_to=str(data_root), train=True)
    test_ds = tonic.datasets.SHD(save_to=str(data_root), train=False)

    def convert(ds):
        times = []
        units = []
        labels = []
        for i in range(len(ds)):
            events, label = ds[i]
            # 수정됨: events는 structured array이므로 필드명으로 접근해야 함
            # 't': timestamps, 'x': neuron index
            times.append(events["t"].astype(np.float32))
            units.append(events["x"].astype(np.int64))
            labels.append(int(label))
        return {"times": np.array(times, dtype=object), "units": np.array(units, dtype=object)}, np.array(labels)

    x_train, y_train = convert(train_ds)
    x_test, y_test = convert(test_ds)
    return {"times": x_train["times"], "units": x_train["units"], "labels": y_train}, {
        "times": x_test["times"],
        "units": x_test["units"],
        "labels": y_test,
    }


class SpikeIterator:
    def __init__(
        self,
        X: dict,
        y: np.ndarray,
        batch_size: int,
        nb_steps: int,
        nb_units: int,
        max_time: float,
        device: torch.device,
        shuffle: bool = True,
    ):
        self.batch_size = batch_size
        self.nb_steps = nb_steps
        self.nb_units = nb_units
        self.shuffle = shuffle
        self.labels_ = np.array(y, dtype=np.int64)
        self.num_samples = len(self.labels_)
        self.number_of_batches = int(np.ceil(self.num_samples / self.batch_size))
        self.sample_index = np.arange(len(self.labels_))
        self.firing_times = X["times"]
        self.units_fired = X["units"]
        self.time_bins = np.linspace(0, max_time, num=nb_steps)
        self.device = device
        self.reset()

    def reset(self):
        if self.shuffle:
            np.random.shuffle(self.sample_index)
        self.counter = 0

    def __iter__(self):
        return self

    def __len__(self):
        return self.number_of_batches

    def __next__(self):
        if self.counter < self.number_of_batches:
            batch_index = self.sample_index[
                self.batch_size * self.counter : min(self.batch_size * (self.counter + 1), self.num_samples)
            ]
            coo = [[] for _ in range(3)]
            for bc, idx in enumerate(batch_index):
                times = np.digitize(self.firing_times[idx], self.time_bins)
                units = self.units_fired[idx]
                batch = [bc for _ in range(len(times))]
                coo[0].extend(batch)
                coo[1].extend(times)
                coo[2].extend(units)

            indices = torch.LongTensor(coo).to(self.device)
            values = torch.ones(len(coo[0]), device=self.device)
            # Sparse tensor construction
            dense = torch.sparse_coo_tensor(indices, values, torch.Size([len(batch_index), self.nb_steps, self.nb_units]))
            X_batch = dense.to_dense()
            y_batch = torch.tensor(self.labels_[batch_index], device=self.device)
            self.counter += 1
            return X_batch, y_batch
        raise StopIteration


def get_shd_loaders(
    batch_size: int,
    T: int = 250,
    max_time: float = 1.4,
    in_dim: int = 700,
    seed: int = 0,
    num_workers: int = 0,
    data_root: str | Path | None = None,
    cache_dense: bool = True,
    device: torch.device | None = None,
):
    np.random.seed(seed)
    resolved_root = Path(data_root) if data_root is not None else DEFAULT_DATA_ROOT
    data_root = resolved_root
    cache_root = data_root / "cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_file = cache_root / f"shd_cache_T{T}_mt{max_time}_in{in_dim}.npz"

    loaded_cache = None
    if cache_dense and cache_file.exists():
        try:
            loaded_cache = np.load(cache_file, allow_pickle=True)
            x_train = {"times": loaded_cache["train_times"], "units": loaded_cache["train_units"]}
            y_train = loaded_cache["train_labels"]
            x_test = {"times": loaded_cache["test_times"], "units": loaded_cache["test_units"]}
            y_test = loaded_cache["test_labels"]
        except KeyError:
            cache_file.unlink(missing_ok=True)
            loaded_cache = None

    if loaded_cache is None:
        train_raw, test_raw = _load_shd_raw(data_root)
        x_train = {"times": train_raw["times"], "units": train_raw["units"]}
        y_train = train_raw["labels"]
        x_test = {"times": test_raw["times"], "units": test_raw["units"]}
        y_test = test_raw["labels"]
        if cache_dense:
            np.savez(
                cache_file,
                train_times=train_raw["times"],
                train_units=train_raw["units"],
                train_labels=y_train,
                test_times=test_raw["times"],
                test_units=test_raw["units"],
                test_labels=y_test,
            )

    device = device or torch.device("cpu")
    train_loader = SpikeIterator(x_train, y_train, batch_size, T, in_dim, max_time, device=device, shuffle=True)
    test_loader = SpikeIterator(x_test, y_test, batch_size, T, in_dim, max_time, device=device, shuffle=False)
    return train_loader, test_loader
