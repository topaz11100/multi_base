import os
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_ROOT = os.path.abspath(os.path.join(THIS_DIR, "../../.."))
sys.path.insert(0, PROJ_ROOT)

from src.common.datasets import get_shd_loaders  # noqa: E402


def load_data(
    data_root: str,
    batch_size: int,
    num_workers: int,
    download: bool,
    *,
    T_event: int = 250,
    seed=None,
):
    return get_shd_loaders(
        data_root,
        batch_size=batch_size,
        num_workers=num_workers,
        download=download,
        T=int(T_event),
        seed=seed,
    )
