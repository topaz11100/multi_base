from __future__ import annotations

import gzip
import os
import random
import pickle
import shutil
import struct
import tarfile
import time
import sys
import urllib.error
import urllib.request
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from tqdm.auto import tqdm

# Optional torchvision (MNIST/CIFAR10 can be simplified when torchvision is available).
# If torchvision is missing or fails to import (e.g., mismatched binary), we fall back to
# the minimal built-in loaders below.
try:  # pragma: no cover
    from torchvision import datasets as tv_datasets  # type: ignore
    from torchvision import transforms as tv_transforms  # type: ignore
    _HAS_TORCHVISION = True
    _TORCHVISION_IMPORT_ERROR = None
except Exception as _e:  # pragma: no cover
    tv_datasets = None
    tv_transforms = None
    _HAS_TORCHVISION = False
    _TORCHVISION_IMPORT_ERROR = _e

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .fft_analysis import rfft_log_mag, bin_spectrum


# ----------------------------------------------------------------------------
# DataLoader seeding helpers
# ----------------------------------------------------------------------------

def _make_worker_init_fn(seed: int):
    """Initialize python/numpy/torch RNG per worker deterministically."""
    def _fn(worker_id: int):
        s = int(seed) + int(worker_id)
        random.seed(s)
        np.random.seed(s)
        torch.manual_seed(s)
    return _fn



# -----------------------------------------------------------------------------
# Download helpers (used for SHD/SSC; MNIST/CIFAR10 prefer torchvision when available)
# -----------------------------------------------------------------------------

def _download(url: str, dst_path: str) -> None:
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    if os.path.exists(dst_path):
        return
    tmp = dst_path + ".tmp"

    # Remove stale tmp to avoid confusing partial files.
    if os.path.exists(tmp):
        try:
            os.remove(tmp)
        except OSError:
            pass

    tqdm.write(f"Downloading: {url} -> {dst_path}")

    # NOTE:
    # - We avoid urllib.request.urlretrieve() here because it has poor progress reporting
    #   and can appear to "hang" in nohup logs.
    # - urlopen(timeout=...) ensures we fail fast on offline clusters.
    req = urllib.request.Request(
        url,
        headers={
            # Some hosts may reject the default Python user-agent.
            "User-Agent": "Mozilla/5.0 (compatible; multi_base/1.0)"
        },
    )

    timeout_sec = 60
    chunk_size = 256 * 1024  # 256KB
    last_print = 0.0

    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            total = resp.headers.get("Content-Length")
            total_bytes = int(total) if (total is not None and str(total).isdigit()) else None

            # For interactive runs, use tqdm progress bar. For nohup logs, print
            # periodic progress lines to avoid excessive log spam.
            use_pbar = hasattr(sys.stderr, "isatty") and sys.stderr.isatty()
            pbar = None
            if use_pbar:
                pbar = tqdm(total=total_bytes, unit="B", unit_scale=True, unit_divisor=1024)

            downloaded = 0
            with open(tmp, "wb") as f:
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if pbar is not None:
                        pbar.update(len(chunk))
                    else:
                        now = time.time()
                        if now - last_print >= 10:
                            if total_bytes:
                                pct = 100.0 * float(downloaded) / float(total_bytes)
                                tqdm.write(
                                    f"  ... {downloaded/1024/1024:.1f}MB / {total_bytes/1024/1024:.1f}MB ({pct:.1f}%)"
                                )
                            else:
                                tqdm.write(f"  ... {downloaded/1024/1024:.1f}MB")
                            last_print = now

            if pbar is not None:
                pbar.close()

        os.replace(tmp, dst_path)
    except Exception:
        # Clean up tmp to avoid future "resume" confusion.
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
        raise


def _download_with_fallback(urls, dst_path: str) -> None:
    last_err = None
    for url in urls:
        try:
            _download(url, dst_path)
            return
        except Exception as e:
            last_err = e
            if os.path.exists(dst_path):
                return
    raise RuntimeError(f"Failed to download {dst_path}. Last error: {last_err}")


def _extract_tar_gz(tar_gz_path: str, dst_dir: str) -> None:
    marker = os.path.join(dst_dir, ".extracted")
    if os.path.exists(marker):
        return
    os.makedirs(dst_dir, exist_ok=True)
    tqdm.write(f"Extracting: {tar_gz_path} -> {dst_dir}")
    with tarfile.open(tar_gz_path, "r:gz") as tar:
        tar.extractall(path=dst_dir)
    with open(marker, "w", encoding="utf-8") as f:
        f.write("ok")


# -----------------------------------------------------------------------------
# MNIST (IDX format)
# -----------------------------------------------------------------------------

MNIST_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}

MNIST_URLS = [
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
    "http://yann.lecun.com/exdb/mnist/",
]


def _read_idx_images(gz_path: str) -> np.ndarray:
    with gzip.open(gz_path, "rb") as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"Invalid MNIST image file magic {magic} in {gz_path}")
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data.reshape(num, rows, cols)


def _read_idx_labels(gz_path: str) -> np.ndarray:
    with gzip.open(gz_path, "rb") as f:
        magic, num = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"Invalid MNIST label file magic {magic} in {gz_path}")
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data.reshape(num)


class MNISTRaw(Dataset):
    def __init__(self, root: str, train: bool, download: bool = True):
        self.root = root
        self.train = bool(train)
        os.makedirs(self.root, exist_ok=True)

        if download:
            for key, fname in MNIST_FILES.items():
                dst = os.path.join(self.root, fname)
                if not os.path.exists(dst):
                    urls = [base + fname for base in MNIST_URLS]
                    _download_with_fallback(urls, dst)

        if self.train:
            img_path = os.path.join(self.root, MNIST_FILES["train_images"])
            lbl_path = os.path.join(self.root, MNIST_FILES["train_labels"])
        else:
            img_path = os.path.join(self.root, MNIST_FILES["test_images"])
            lbl_path = os.path.join(self.root, MNIST_FILES["test_labels"])

        self.images = _read_idx_images(img_path)  # (N,28,28) uint8
        self.labels = _read_idx_labels(lbl_path)  # (N,) uint8

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, idx: int):
        x = self.images[idx].astype(np.float32) / 255.0  # (28,28)
        y = int(self.labels[idx])
        return torch.from_numpy(x).unsqueeze(0), y  # (1,28,28)


# -----------------------------------------------------------------------------
# CIFAR-10 (python pickle batches)
# -----------------------------------------------------------------------------

CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"


def _load_cifar_batch(path: str) -> Tuple[np.ndarray, np.ndarray]:
    with open(path, "rb") as f:
        d = pickle.load(f, encoding="bytes")
    data = d[b"data"]  # (N,3072)
    labels = d.get(b"labels", d.get(b"fine_labels"))
    x = data.reshape(-1, 3, 32, 32)
    y = np.array(labels, dtype=np.int64)
    return x, y


class CIFAR10Raw(Dataset):
    def __init__(self, root: str, train: bool, download: bool = True, normalize: bool = True, augment: bool = False):
        self.root = root
        self.train = bool(train)
        self.normalize = bool(normalize)
        self.augment = bool(augment)
        os.makedirs(self.root, exist_ok=True)

        tar_path = os.path.join(self.root, "cifar-10-python.tar.gz")
        extract_dir = os.path.join(self.root, "cifar-10-batches-py")
        if download and not os.path.exists(extract_dir):
            _download(CIFAR10_URL, tar_path)
            _extract_tar_gz(tar_path, self.root)

        if not os.path.exists(extract_dir):
            raise FileNotFoundError(
                f"CIFAR-10 not found in {extract_dir}. Set download=True or place extracted folder there."
            )

        if self.train:
            xs, ys = [], []
            for i in range(1, 6):
                x, y = _load_cifar_batch(os.path.join(extract_dir, f"data_batch_{i}"))
                xs.append(x)
                ys.append(y)
            self.images = np.concatenate(xs, axis=0)  # (50000,3,32,32)
            self.labels = np.concatenate(ys, axis=0)
        else:
            self.images, self.labels = _load_cifar_batch(os.path.join(extract_dir, "test_batch"))

        # Normalization constants (common CIFAR-10)
        self.mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
        self.std = torch.tensor([0.2470, 0.2435, 0.2616]).view(3, 1, 1)

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def _random_crop(self, x: torch.Tensor, pad: int = 4) -> torch.Tensor:
        # x: (3,32,32)
        if pad <= 0:
            return x
        x_p = torch.nn.functional.pad(x, (pad, pad, pad, pad), mode="constant", value=0.0)
        _, H, W = x_p.shape
        top = torch.randint(0, H - 32 + 1, (1,)).item()
        left = torch.randint(0, W - 32 + 1, (1,)).item()
        return x_p[:, top : top + 32, left : left + 32]

    def _random_hflip(self, x: torch.Tensor, p: float = 0.5) -> torch.Tensor:
        if torch.rand(()) < p:
            return torch.flip(x, dims=[2])
        return x

    def __getitem__(self, idx: int):
        x = torch.from_numpy(self.images[idx].astype(np.float32) / 255.0)  # (3,32,32)
        y = int(self.labels[idx])

        if self.train and self.augment:
            x = self._random_crop(x, pad=4)
            x = self._random_hflip(x, p=0.5)

        if self.normalize:
            x = (x - self.mean) / self.std

        return x, y


def _ensure_mnist_torchvision_layout(dataset_dir: str) -> None:
    """
    Torchvision's MNIST expects raw .gz files under: <dataset_dir>/raw/.

    This project historically stored MNIST *.gz directly under <dataset_dir> (legacy layout).
    To avoid re-downloading the same data, we migrate those files into <dataset_dir>/raw/
    if needed.
    """
    try:
        os.makedirs(dataset_dir, exist_ok=True)
        raw_dir = os.path.join(dataset_dir, 'raw')
        os.makedirs(raw_dir, exist_ok=True)
        for fname in MNIST_FILES.values():
            src = os.path.join(dataset_dir, fname)
            dst = os.path.join(raw_dir, fname)
            if os.path.exists(src) and (not os.path.exists(dst)):
                try:
                    os.replace(src, dst)
                except OSError:
                    shutil.copy2(src, dst)
                    os.remove(src)
    except Exception:
        # Best-effort migration only. If this fails, torchvision will still be able
        # to download when download=True.
        return


# -----------------------------------------------------------------------------
# Sequential wrappers
# -----------------------------------------------------------------------------


class SequentialMNIST(Dataset):
    """MNIST -> sequence of length 784 with input_dim=1."""

    def __init__(self, root: str, train: bool, download: bool = True):
        # Prefer torchvision's dataset implementation when available.
        if _HAS_TORCHVISION and tv_datasets is not None and tv_transforms is not None:
            # Project layout uses dataset-specific folder: <data_root>/MNIST/
            tv_root = os.path.abspath(os.path.join(root, os.pardir))
            dataset_dir = os.path.join(tv_root, "MNIST")
            _ensure_mnist_torchvision_layout(dataset_dir)

            # Torchvision MNIST requires processed .pt files; if only raw .gz exists, we
            # enable download=True to trigger *processing* without re-downloading.
            processed_train = os.path.join(dataset_dir, "processed", "training.pt")
            processed_test = os.path.join(dataset_dir, "processed", "test.pt")
            processed_ok = os.path.exists(processed_train) and os.path.exists(processed_test)
            raw_ok = all(os.path.exists(os.path.join(dataset_dir, "raw", f)) for f in MNIST_FILES.values())
            tv_download = bool(download) or ((not processed_ok) and raw_ok)

            self.base = tv_datasets.MNIST(
                root=tv_root,
                train=bool(train),
                download=tv_download,
                transform=tv_transforms.ToTensor(),
            )
        else:
            # Fallback: minimal pure-Python loader (no torchvision dependency).
            self.base = MNISTRaw(root=root, train=train, download=download)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        x, y = self.base[idx]  # x: (1,28,28)
        x = x.view(1, -1).transpose(0, 1).contiguous()  # (784,1)
        return x.to(torch.float32), int(y)


class SequentialCIFAR10(Dataset):
    """
    CIFAR10 -> sequence.
    mode:
      - "parallel": T=1024, input_dim=3 (RGB vector per pixel)
      - "serial":   T=3072, input_dim=1 (R then G then B)
    """

    def __init__(self, root: str, train: bool, download: bool = True, mode: str = "parallel"):
        self.mode = mode
        # Prefer torchvision for CIFAR10 download/decoding/augmentation when available.
        if _HAS_TORCHVISION and tv_datasets is not None and tv_transforms is not None:
            mean = [0.4914, 0.4822, 0.4465]
            std = [0.2470, 0.2435, 0.2616]
            if bool(train):
                tfm = tv_transforms.Compose(
                    [
                        tv_transforms.RandomCrop(32, padding=4),
                        tv_transforms.RandomHorizontalFlip(),
                        tv_transforms.ToTensor(),
                        tv_transforms.Normalize(mean, std),
                    ]
                )
            else:
                tfm = tv_transforms.Compose(
                    [
                        tv_transforms.ToTensor(),
                        tv_transforms.Normalize(mean, std),
                    ]
                )
            self.base = tv_datasets.CIFAR10(
                root=root,
                train=bool(train),
                download=bool(download),
                transform=tfm,
            )
        else:
            # Fallback: minimal pure-Python loader (no torchvision dependency).
            if not _HAS_TORCHVISION and (_TORCHVISION_IMPORT_ERROR is not None):
                tqdm.write(
                    f"[WARN] torchvision import failed ({type(_TORCHVISION_IMPORT_ERROR).__name__}: "
                    f"{_TORCHVISION_IMPORT_ERROR}). Using built-in CIFAR10 loader."
                )
            self.base = CIFAR10Raw(root=root, train=train, download=download, normalize=True, augment=train)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        x, y = self.base[idx]  # x: (3,32,32)
        if self.mode == "parallel":
            x = x.permute(1, 2, 0).contiguous().view(-1, 3)  # (1024,3)
        elif self.mode == "serial":
            x = x.contiguous().view(-1, 1)  # (3072,1)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
        return x.to(torch.float32), int(y)


# -----------------------------------------------------------------------------
# SHD / SSC (h5)
# -----------------------------------------------------------------------------

class EventH5Dataset(Dataset):
    """
    SHD/SSC-style event dataset stored in an HDF5 file with groups:
      - spikes/times : vlen float (seconds)
      - spikes/units : vlen int
      - labels       : int

    ⚠️ Preprocessing equivalence (Origin/ verification)
    ---------------------------------------------------
    The default settings of this class are designed to be *semantically aligned* with the
    author-provided preprocessing scripts:

      - Origin/DH-SNN-main/SHD/shd_generate_dataset.py
      - Origin/DH-SNN-main/SSC/ssc_generate_dataset.py

    Concretely, when using:
      - max_time = 1.0
      - binning = "origin"        (time -> bin via i = ceil(t/dt) with i in [0, T-1])
      - channel_flip = True       (reverse channel order)
      - unit_indexing = "auto"    (infer 0-based vs 1-based unit IDs)
      - align_to_first_event = False
      - use_event_counts = False  (binary frames)

    ...the resulting (T, 700) binary frame sequence matches the intent of the Origin code
    while being vectorized (fast) and stable (no per-sample indexing heuristics).

    Parameters
    ----------
    h5_path:
        Path to *.h5.
    T:
        Number of time bins / steps.
    num_units:
        Number of input channels (SHD/SSC: 700).
    max_time:
        Window length in seconds. Origin preprocessing uses 1.0s.
    binning:
        - "origin": assign each event to the *earliest* bin i such that t <= i*dt.
                    This is equivalent to i = ceil(t/dt), and drops events with i >= T.
        - "floor":  standard left-closed binning i = floor(t/dt) with i in [0, T-1].
    unit_indexing:
        - "auto": infer whether raw unit IDs are 0-based (0..699) or 1-based (1..700)
                  by probing multiple samples at dataset construction time.
        - "0": force 0-based
        - "1": force 1-based
    channel_flip:
        If True, reverse channel order (Origin uses vector[700-vals] = 1).
    align_to_first_event:
        If True, shift times so that the first event in each sample starts at t=0.
        (Origin code does NOT do this; keep False for equivalence.)
    use_event_counts:
        If True, accumulate counts per (bin, channel). Origin uses binary frames, so
        keep False for equivalence.
    """

    _ALL_KEYS = ("dendrite_input", "dendrite_state", "soma_input", "soma_state", "output")

    # Cache inferred unit offsets per file (avoid repeated probing).
    _UNIT_OFFSET_CACHE = {}

    def __init__(
        self,
        h5_path: str,
        T: int = 250,
        num_units: int = 700,
        *,
        max_time: float = 1.0,
        binning: str = "origin",
        unit_indexing: str = "auto",
        channel_flip: bool = True,
        align_to_first_event: bool = False,
        use_event_counts: bool = False,
        probe_units: int = 2048,
    ):
        self.h5_path = str(h5_path)
        self.T = int(T)
        self.num_units = int(num_units)

        self.max_time = float(max_time)
        if self.T <= 0:
            raise ValueError(f"T must be >= 1, got {self.T}")
        if self.num_units <= 0:
            raise ValueError(f"num_units must be >= 1, got {self.num_units}")
        if not (self.max_time > 0.0):
            raise ValueError(f"max_time must be > 0, got {self.max_time}")

        self.dt = self.max_time / float(self.T)

        self.binning = str(binning).lower().strip()
        if self.binning not in ("origin", "floor"):
            raise ValueError(f"Unsupported binning={binning!r}. Use 'origin' or 'floor'.")

        self.channel_flip = bool(channel_flip)
        self.align_to_first_event = bool(align_to_first_event)
        self.use_event_counts = bool(use_event_counts)

        unit_indexing = str(unit_indexing).lower().strip()
        if unit_indexing in ("auto", "a"):
            self.unit_offset = self._infer_unit_offset(self.h5_path, self.num_units, probe=int(probe_units))
        elif unit_indexing in ("0", "0-based", "0based", "zero"):
            self.unit_offset = 0
        elif unit_indexing in ("1", "1-based", "1based", "one"):
            self.unit_offset = 1
        else:
            raise ValueError(f"Unsupported unit_indexing={unit_indexing!r}. Use 'auto', '0', or '1'.")

        # Lazy-open HDF5 handle per worker/process.
        self._h5 = None
        self._len = None

    # ------------------------------------------------------------------
    # HDF5 helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _infer_unit_offset(h5_path: str, num_units: int, probe: int = 2048) -> int:
        """
        Infer whether unit IDs are 0-based (0..num_units-1) or 1-based (1..num_units).

        We avoid per-sample heuristics (e.g., `min==1`) because they can silently break
        channel consistency across samples (catastrophic for learning).
        """
        key = (os.path.abspath(h5_path), int(num_units))
        if key in EventH5Dataset._UNIT_OFFSET_CACHE:
            return int(EventH5Dataset._UNIT_OFFSET_CACHE[key])

        try:
            import h5py  # type: ignore
        except Exception:
            # Fall back to safest assumption (0-based).
            EventH5Dataset._UNIT_OFFSET_CACHE[key] = 0
            return 0

        if not os.path.exists(h5_path):
            EventH5Dataset._UNIT_OFFSET_CACHE[key] = 0
            return 0

        saw_zero = False
        saw_num_units = False
        global_min = None
        global_max = None

        with h5py.File(h5_path, "r") as f:
            u_ds = f["spikes"]["units"]
            n = int(u_ds.shape[0])
            if n <= 0:
                EventH5Dataset._UNIT_OFFSET_CACHE[key] = 0
                return 0
            k = int(min(max(1, probe), n))
            # Evenly spaced probes -> more robust than first-k only.
            idxs = np.linspace(0, n - 1, num=k, dtype=np.int64)
            for i in idxs:
                u = np.asarray(u_ds[int(i)], dtype=np.int64)
                if u.size == 0:
                    continue
                if (u == 0).any():
                    saw_zero = True
                if (u == num_units).any():
                    saw_num_units = True
                mi = int(u.min())
                ma = int(u.max())
                global_min = mi if global_min is None else min(global_min, mi)
                global_max = ma if global_max is None else max(global_max, ma)

        # Strong signals.
        if saw_num_units:
            EventH5Dataset._UNIT_OFFSET_CACHE[key] = 1
            return 1
        if saw_zero:
            EventH5Dataset._UNIT_OFFSET_CACHE[key] = 0
            return 0

        # Weak signal: if everything we saw is within [1, num_units], treat as 1-based.
        # (For typical SHD/SSC downloads, seeing a 0 across ~2k samples is very likely if 0-based.)
        if global_min is not None and global_max is not None:
            if global_min >= 1 and global_max <= num_units:
                EventH5Dataset._UNIT_OFFSET_CACHE[key] = 1
                return 1

        EventH5Dataset._UNIT_OFFSET_CACHE[key] = 0
        return 0

    def _ensure_open(self):
        if self._h5 is None:
            import h5py  # type: ignore

            self._h5 = h5py.File(self.h5_path, "r")
            self._times = self._h5["spikes"]["times"]
            self._units = self._h5["spikes"]["units"]
            self._labels = self._h5["labels"]
            self._len = int(self._labels.shape[0])

    def __len__(self) -> int:
        if self._len is None:
            self._ensure_open()
        return int(self._len)

    def __getitem__(self, idx: int):
        self._ensure_open()
        times = np.asarray(self._times[int(idx)], dtype=np.float32)  # seconds
        units = np.asarray(self._units[int(idx)], dtype=np.int64)
        label = int(np.asarray(self._labels[int(idx)]).item())

        x = np.zeros((self.T, self.num_units), dtype=np.float32)

        if times.size == 0 or units.size == 0:
            return torch.from_numpy(x), label

        # Optional alignment (NOT used in Origin preprocessing).
        if self.align_to_first_event:
            t0 = float(times.min())
            times = times - t0

        # Convert units to 0-based indexing using the inferred offset.
        units0 = units.astype(np.int64) - int(self.unit_offset)

        # Validity mask: units within [0, num_units-1]
        m_u = (units0 >= 0) & (units0 < self.num_units)
        if not np.any(m_u):
            return torch.from_numpy(x), label
        units0 = units0[m_u]
        times = times[m_u]

        # Clamp negative times to 0 for origin-style thresholding.
        t = np.maximum(times.astype(np.float32), 0.0)

        # Origin-style time binning: i = ceil(t/dt), i in [0, T-1]
        if self.binning == "origin":
            bin_idx = np.ceil(t / float(self.dt)).astype(np.int64)
        else:  # "floor"
            bin_idx = np.floor(t / float(self.dt)).astype(np.int64)

        m_t = (bin_idx >= 0) & (bin_idx < self.T)
        if not np.any(m_t):
            return torch.from_numpy(x), label
        bin_idx = bin_idx[m_t]
        units0 = units0[m_t]

        # Channel mapping (Origin uses reversed index).
        if self.channel_flip:
            ch = (self.num_units - 1) - units0
        else:
            ch = units0

        # Fill dense frame tensor.
        if self.use_event_counts:
            np.add.at(x, (bin_idx, ch), 1.0)
        else:
            x[bin_idx, ch] = 1.0

        return torch.from_numpy(x), label


def ensure_shd_ssc_files(data_root: str, dataset: str, download: bool = True) -> Tuple[str, str, Optional[str]]:
    """
    Ensure dataset files exist in data_root/<dataset>/.
    Returns (train_path, test_path, valid_path).
    """
    dataset = dataset.upper()
    ddir = os.path.join(data_root, dataset)
    os.makedirs(ddir, exist_ok=True)

    if dataset == "SHD":
        train_h5 = os.path.join(ddir, "shd_train.h5")
        test_h5 = os.path.join(ddir, "shd_test.h5")
        valid_h5 = None

        if download and (not os.path.exists(train_h5) or not os.path.exists(test_h5)):
            base = "https://zenkelab.org/datasets/"
            train_gz = os.path.join(ddir, "shd_train.h5.gz")
            test_gz = os.path.join(ddir, "shd_test.h5.gz")
            try:
                if not os.path.exists(train_h5):
                    _download(base + "shd_train.h5.gz", train_gz)
                    _gunzip(train_gz, train_h5)
                if not os.path.exists(test_h5):
                    _download(base + "shd_test.h5.gz", test_gz)
                    _gunzip(test_gz, test_h5)
            except Exception as e:
                tqdm.write(f"[WARN] Automatic download failed: {e}")
                tqdm.write(f"Please download SHD files manually into: {ddir}")
        return train_h5, test_h5, valid_h5

    if dataset == "SSC":
        train_h5 = os.path.join(ddir, "ssc_train.h5")
        test_h5 = os.path.join(ddir, "ssc_test.h5")
        valid_h5 = os.path.join(ddir, "ssc_valid.h5")

        if download and (not os.path.exists(train_h5) or not os.path.exists(test_h5)):
            base = "https://zenkelab.org/datasets/"
            train_gz = os.path.join(ddir, "ssc_train.h5.gz")
            test_gz = os.path.join(ddir, "ssc_test.h5.gz")
            valid_gz = os.path.join(ddir, "ssc_valid.h5.gz")
            try:
                if not os.path.exists(train_h5):
                    _download(base + "ssc_train.h5.gz", train_gz)
                    _gunzip(train_gz, train_h5)
                if not os.path.exists(test_h5):
                    _download(base + "ssc_test.h5.gz", test_gz)
                    _gunzip(test_gz, test_h5)
                if not os.path.exists(valid_h5):
                    _download(base + "ssc_valid.h5.gz", valid_gz)
                    _gunzip(valid_gz, valid_h5)
            except Exception as e:
                tqdm.write(f"[WARN] Automatic download failed: {e}")
                tqdm.write(f"Please download SSC files manually into: {ddir}")
        return train_h5, test_h5, valid_h5

    raise ValueError(f"Unknown dataset: {dataset}")


# -----------------------------------------------------------------------------
# DataLoader helpers
# -----------------------------------------------------------------------------

def get_smnist_loaders(
    data_root: str,
    batch_size: int = 128,
    num_workers: int = 4,
    download: bool = True,
    seed: Optional[int] = None,
):
    root = os.path.join(data_root, "MNIST")
    train_ds = SequentialMNIST(root=root, train=True, download=download)
    test_ds = SequentialMNIST(root=root, train=False, download=download)

    g_train = None
    g_test = None
    worker_init_fn = None
    if seed is not None:
        g_train = torch.Generator()
        g_train.manual_seed(int(seed))
        g_test = torch.Generator()
        g_test.manual_seed(int(seed) + 1)
        worker_init_fn = _make_worker_init_fn(int(seed))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=torch.cuda.is_available(), generator=g_train, worker_init_fn=worker_init_fn, persistent_workers=(num_workers > 0))
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available(), generator=g_test, worker_init_fn=worker_init_fn, persistent_workers=(num_workers > 0))
    return train_loader, test_loader, 10, 1, 784


def get_scifar10_loaders(
    data_root: str,
    batch_size: int = 128,
    num_workers: int = 4,
    download: bool = True,
    mode: str = "parallel",
    seed: Optional[int] = None,
):
    root = os.path.join(data_root, "CIFAR10")
    train_ds = SequentialCIFAR10(root=root, train=True, download=download, mode=mode)
    test_ds = SequentialCIFAR10(root=root, train=False, download=download, mode=mode)

    g_train = None
    g_test = None
    worker_init_fn = None
    if seed is not None:
        g_train = torch.Generator()
        g_train.manual_seed(int(seed))
        g_test = torch.Generator()
        g_test.manual_seed(int(seed) + 1)
        worker_init_fn = _make_worker_init_fn(int(seed))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=torch.cuda.is_available(), generator=g_train, worker_init_fn=worker_init_fn, persistent_workers=(num_workers > 0))
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available(), generator=g_test, worker_init_fn=worker_init_fn, persistent_workers=(num_workers > 0))
    input_dim = 3 if mode == "parallel" else 1
    T = 1024 if mode == "parallel" else 3072
    return train_loader, test_loader, 10, input_dim, T


def get_shd_loaders(
    data_root: str,
    batch_size: int = 128,
    num_workers: int = 4,
    download: bool = True,
    T: int = 250,
    seed: Optional[int] = None,
):
    train_h5, test_h5, _ = ensure_shd_ssc_files(data_root, "SHD", download=download)
    train_ds = EventH5Dataset(train_h5, T=T, num_units=700, max_time=1.0, binning="origin", unit_indexing="auto", channel_flip=True, align_to_first_event=False, use_event_counts=False)
    test_ds = EventH5Dataset(test_h5, T=T, num_units=700, max_time=1.0, binning="origin", unit_indexing="auto", channel_flip=True, align_to_first_event=False, use_event_counts=False)

    g_train = None
    g_test = None
    worker_init_fn = None
    if seed is not None:
        g_train = torch.Generator()
        g_train.manual_seed(int(seed))
        g_test = torch.Generator()
        g_test.manual_seed(int(seed) + 1)
        worker_init_fn = _make_worker_init_fn(int(seed))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=torch.cuda.is_available(), generator=g_train, worker_init_fn=worker_init_fn, persistent_workers=(num_workers > 0))
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available(), generator=g_test, worker_init_fn=worker_init_fn, persistent_workers=(num_workers > 0))
    return train_loader, test_loader, 20, 700, T


def get_ssc_loaders(
    data_root: str,
    batch_size: int = 128,
    num_workers: int = 4,
    download: bool = True,
    T: int = 250,
    use_valid_as_test: bool = False,
    seed: Optional[int] = None,
):
    train_h5, test_h5, valid_h5 = ensure_shd_ssc_files(data_root, "SSC", download=download)
    train_ds = EventH5Dataset(train_h5, T=T, num_units=700, max_time=1.0, binning="origin", unit_indexing="auto", channel_flip=True, align_to_first_event=False, use_event_counts=False)
    if use_valid_as_test and valid_h5 is not None and os.path.exists(valid_h5):
        test_ds = EventH5Dataset(valid_h5, T=T, num_units=700, max_time=1.0, binning="origin", unit_indexing="auto", channel_flip=True, align_to_first_event=False, use_event_counts=False)
    else:
        test_ds = EventH5Dataset(test_h5, T=T, num_units=700, max_time=1.0, binning="origin", unit_indexing="auto", channel_flip=True, align_to_first_event=False, use_event_counts=False)

    g_train = None
    g_test = None
    worker_init_fn = None
    if seed is not None:
        g_train = torch.Generator()
        g_train.manual_seed(int(seed))
        g_test = torch.Generator()
        g_test.manual_seed(int(seed) + 1)
        worker_init_fn = _make_worker_init_fn(int(seed))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=torch.cuda.is_available(), generator=g_train, worker_init_fn=worker_init_fn, persistent_workers=(num_workers > 0))
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available(), generator=g_test, worker_init_fn=worker_init_fn, persistent_workers=(num_workers > 0))
    return train_loader, test_loader, 35, 700, T


# -----------------------------------------------------------------------------
# Visualization helpers for freq_analysis
# -----------------------------------------------------------------------------

def visualize_input_sequence(
    dataset: str,
    x_seq: torch.Tensor,
    out_dir: str,
    fft_band_edges=None,
    fft_band_reduce: str = "mean",
    title_prefix: str = "",
) -> None:
    """
    Save:
      - image.png: spatial/raster visualization
      - image_fft.png: exact rFFT spectrum of an aggregated 1D signal
      - image_fft_band.png: binned version if fft_band_edges is not None
    """
    os.makedirs(out_dir, exist_ok=True)

    x = x_seq.detach().cpu().to(torch.float32)
    dataset_u = dataset.upper()

    # image.png
    if dataset_u in ("S-MNIST", "SMNIST", "MNIST"):
        img = x.view(28, 28).numpy()
        plt.figure(figsize=(3.2, 3.2))
        plt.imshow(img, cmap="gray", interpolation="nearest")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "image.png"), dpi=200)
        plt.close()
        agg = x.view(-1).numpy()
    elif dataset_u in ("S-CIFAR10", "SCIFAR10", "CIFAR10"):
        if x.shape[1] == 3 and x.shape[0] == 1024:
            img = x.view(32, 32, 3).numpy()
            img_min = img.min()
            img_max = img.max()
            if img_max > img_min:
                img = (img - img_min) / (img_max - img_min)
            plt.figure(figsize=(3.4, 3.4))
            plt.imshow(img, interpolation="nearest")
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "image.png"), dpi=200)
            plt.close()
            agg = x.mean(dim=1).numpy()
        else:
            agg = x.view(-1).numpy()
            plt.figure(figsize=(3.4, 2.2))
            plt.plot(agg, linewidth=1.0)
            plt.grid(True, which="both", alpha=0.28)
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "image.png"), dpi=200)
            plt.close()
    else:
        # SHD / SSC: raster (time x channel)
        mat = x.numpy().T  # (C,T)
        plt.figure(figsize=(6.2, 3.2))
        plt.imshow(mat, aspect="auto", interpolation="nearest")
        plt.xlabel("t")
        plt.ylabel("unit")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "image.png"), dpi=200)
        plt.close()
        agg = x.mean(dim=1).numpy()

    # image_fft.png (normalized frequency axis: cycles/step)
    from .fft_analysis import rfft_freqs, band_edges_to_bin_ranges

    freqs = rfft_freqs(len(agg), d=1.0)
    S = rfft_log_mag(agg, dim=-1)  # numpy
    plt.figure(figsize=(6.2, 3.2))
    plt.plot(freqs, S, linewidth=1.2)
    plt.xlabel("frequency (cycles/step)")
    plt.ylabel("log(1+|rFFT|)")
    plt.grid(True, which="both", alpha=0.28)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "image_fft.png"), dpi=200)
    plt.close()

    if fft_band_edges is not None and len(fft_band_edges) > 0:
        ranges = band_edges_to_bin_ranges(len(agg), fft_band_edges, d=1.0)
        Sb = bin_spectrum(S, ranges, dim=-1, reduce=fft_band_reduce)
        centers = [(float(fft_band_edges[i]) + float(fft_band_edges[i + 1])) / 2.0 for i in range(len(fft_band_edges) - 1)]
        plt.figure(figsize=(6.2, 3.2))
        plt.plot(centers, Sb, linewidth=1.2)
        plt.xlabel("frequency band center (cycles/step)")
        plt.ylabel(f"binned ({fft_band_reduce}) log(1+|rFFT|)")
        plt.grid(True, which="both", alpha=0.28)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "image_fft_band.png"), dpi=200)
        plt.close()