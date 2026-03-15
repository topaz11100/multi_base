from __future__ import annotations

import json
import os
import random
import shutil
import math
from dataclasses import asdict, is_dataclass
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np
import torch

try:
    from zoneinfo import ZoneInfo  # py3.9+
except Exception:  # pragma: no cover
    ZoneInfo = None  # type: ignore


SEOUL_TZ = ZoneInfo("Asia/Seoul") if ZoneInfo is not None else None


def now_timestamp_seoul() -> str:
    """Return timestamp string in Asia/Seoul timezone: YYmmdd_HHMMSS."""
    if SEOUL_TZ is None:
        return datetime.now().strftime("%y%m%d_%H%M%S")
    return datetime.now(tz=SEOUL_TZ).strftime("%y%m%d_%H%M%S")


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def project_root_from(__file__: str, up: int = 3) -> str:
    """Return the project root given a file path (typically __file__)."""
    p = os.path.abspath(os.path.dirname(__file__))
    for _ in range(up):
        p = os.path.dirname(p)
    return p


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Determinism (best-effort). Some ops may still be nondeterministic.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False




def get_backend_flags() -> Dict[str, Any]:
    """Return a JSON-serializable snapshot of relevant torch backend flags."""
    return {
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cuda_matmul_allow_tf32": bool(getattr(torch.backends.cuda.matmul, "allow_tf32", False)),
        "cudnn_allow_tf32": bool(getattr(torch.backends.cudnn, "allow_tf32", False)),
        "float32_matmul_precision": (
            str(torch.get_float32_matmul_precision()) if hasattr(torch, "get_float32_matmul_precision") else None
        ),
        "cuda_available": bool(torch.cuda.is_available()),
        "torch_version": str(torch.__version__),
        "cuda_version": str(torch.version.cuda) if hasattr(torch.version, "cuda") else None,
    }


def get_device(device: Optional[str] = None) -> torch.device:
    """CUDA-only device selection.

    This project does **not** support CPU execution (see src/promft.txt).
    Use:
      - device=None or "auto" -> cuda:0
      - device="cuda" -> cuda:0
      - device="cuda:N" -> cuda:N
    """
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for this project (CPU execution is not supported). "
            "Please run on a machine with a CUDA-capable GPU and a CUDA-enabled PyTorch build."
        )

    if device is None or device == "auto":
        return torch.device("cuda:0")

    dev = torch.device(device)
    if dev.type != "cuda":
        raise ValueError(f"Only CUDA devices are supported, got: {device}")
    # Normalize "cuda" -> "cuda:0"
    if dev.index is None:
        return torch.device("cuda:0")
    return dev




# -----------------------------------------------------------------------------
# Formatting helpers
# -----------------------------------------------------------------------------

def float_to_tag(x: float) -> str:
    """Convert a float to a filesystem-friendly tag.

    Examples:
      8.0   -> "8"
      8.25  -> "8p25"
      -0.5  -> "m0p5"
    """
    xf = float(x)
    # Treat near-integers as ints
    if abs(xf - round(xf)) < 1e-9:
        return str(int(round(xf)))
    s = f"{xf:.6f}"  # fixed-point
    s = s.rstrip("0").rstrip(".")
    if s == "-0":
        s = "0"
    # filesystem friendly
    s = s.replace("-", "m").replace(".", "p")
    return s


def derive_branch_from_S_max(S_max: float) -> int:
    """Derive the *tensor-shape* maximum branch count from S_max.

    Project rule (user request): do NOT accept a separate dendritic parameter.
    The maximum branch dimension is derived from S_max only.

    We choose: branch = ceil(S_max)  (with a tiny epsilon for numeric stability)
    """
    s = float(S_max)
    if not (s > 0.0):
        raise ValueError(f"S_max must be > 0, got {S_max}")
    br = int(math.ceil(s - 1e-12))
    return max(1, br)
def to_jsonable(obj: Any) -> Any:
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(x) for x in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    return str(obj)


def save_json(path: str, data: Dict[str, Any], indent: int = 2) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(data), f, indent=indent, ensure_ascii=False)


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_text(path: str, text: str) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def copytree(src: str, dst: str, overwrite: bool = False) -> None:
    if overwrite and os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
