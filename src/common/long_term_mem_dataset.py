from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Dict, Tuple

import numpy as np


def _checksum(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _symbol(rng: np.random.Generator, cls: np.ndarray, length: int, p_low: float, p_high: float, p_noise: float) -> np.ndarray:
    p = np.where(cls > 0, p_high, p_low).astype(np.float32)
    sig = rng.random((cls.shape[0], length), dtype=np.float32) < p[:, None]
    noi = rng.random((cls.shape[0], length), dtype=np.float32) < float(p_noise)
    return np.maximum(sig, noi).astype(np.uint8)


def _build_delayed_split(rng: np.random.Generator, N: int, Ls: int, Ld: int, p_low: float, p_high: float, p_noise: float) -> Dict[str, np.ndarray]:
    reps = N // 4
    rem = N % 4
    pairs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.uint8)
    ab = np.repeat(pairs, reps, axis=0)
    if rem > 0:
        ab = np.concatenate([ab, pairs[:rem]], axis=0)
    rng.shuffle(ab)

    a = ab[:, 0]
    b = ab[:, 1]
    s1 = _symbol(rng, a, Ls, p_low, p_high, p_noise)
    d = (rng.random((N, Ld), dtype=np.float32) < float(p_noise)).astype(np.uint8)
    s2 = _symbol(rng, b, Ls, p_low, p_high, p_noise)
    x = np.concatenate([s1, d, s2], axis=1)[:, :, None]
    T = x.shape[1]
    y = np.bitwise_xor(a, b).astype(np.uint8)
    return {
        "x": x,
        "y": y,
        "eval_idx": np.full((N,), T - 1, dtype=np.int64),
        "signal1_class": a.astype(np.uint8),
        "signal2_class": b.astype(np.uint8),
        "delay_len": np.full((N,), Ld, dtype=np.int64),
    }


def _build_multi_split(rng: np.random.Generator, N: int, Ls: int, Lg: int, K: int, p_low: float, p_high: float, p_noise: float) -> Dict[str, np.ndarray]:
    a = np.tile(np.array([0, 1], dtype=np.uint8), N // 2 + 1)[:N]
    rng.shuffle(a)
    bseq = np.zeros((N, K), dtype=np.uint8)
    for k in range(K):
        col = np.tile(np.array([0, 1], dtype=np.uint8), N // 2 + 1)[:N]
        rng.shuffle(col)
        bseq[:, k] = col

    T = Ls + K * (Lg + Ls)
    x = (rng.random((N, T), dtype=np.float32) < float(p_noise)).astype(np.uint8)
    x[:, :Ls] = np.maximum(x[:, :Ls], _symbol(rng, a, Ls, p_low, p_high, p_noise))
    qidx = np.zeros((N, K), dtype=np.int64)
    for k in range(K):
        st = Ls + k * (Lg + Ls) + Lg
        ed = st + Ls
        x[:, st:ed] = np.maximum(x[:, st:ed], _symbol(rng, bseq[:, k], Ls, p_low, p_high, p_noise))
        qidx[:, k] = ed - 1
    y = np.bitwise_xor(a[:, None], bseq).astype(np.uint8)
    return {
        "x": x[:, :, None],
        "y_seq": y,
        "query_eval_idx": qidx,
        "signal1_class": a,
        "signal2_class_seq": bseq,
        "num_queries": np.full((N,), K, dtype=np.int64),
    }


def _save_npz(path: str, payload: Dict[str, np.ndarray]) -> None:
    np.savez_compressed(path, **payload)


def ensure_serial_xor_datasets(data_root_abs: str, seed: int, p_low: float, p_high: float, p_noise: float, Ls: int, Ld: int, Lg: int, K: int,
                               n_train: int = 50_000, n_val: int = 10_000, n_test: int = 10_000) -> Dict[str, str]:
    root = os.path.abspath(data_root_abs)
    os.makedirs(root, exist_ok=True)
    delayed_dir = os.path.join(root, "delayed_xor_serial")
    multi_dir = os.path.join(root, "multiscale_xor_serial")
    os.makedirs(delayed_dir, exist_ok=True)
    os.makedirs(multi_dir, exist_ok=True)

    rng = np.random.default_rng(int(seed))
    paths: Dict[str, str] = {}
    for name, ddir, fn in [
        ("delayed", delayed_dir, lambda n: _build_delayed_split(rng, n, Ls, Ld, p_low, p_high, p_noise)),
        ("multi", multi_dir, lambda n: _build_multi_split(rng, n, Ls, Lg, K, p_low, p_high, p_noise)),
    ]:
        checksums = {}
        for split, n in [("train", n_train), ("val", n_val), ("test", n_test)]:
            f = os.path.join(ddir, f"{split}.npz")
            if not os.path.isfile(f):
                _save_npz(f, fn(n))
            checksums[split] = _checksum(f)
            paths[f"{name}_{split}"] = f
        meta_path = os.path.join(ddir, "meta.json")
        meta = {
            "dataset": "delayed_xor_serial" if name == "delayed" else "multiscale_xor_serial",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "seed": int(seed),
            "split": {"train": n_train, "val": n_val, "test": n_test},
            "p_low": float(p_low), "p_high": float(p_high), "p_noise": float(p_noise),
            "L_s": int(Ls), "L_g": int(Lg), "L_d": int(Ld), "K": int(K),
            "index_base": 0,
            "checksums": checksums,
            "generator_version": "v2_serial_spec",
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        paths[f"{name}_meta"] = meta_path
    return paths

