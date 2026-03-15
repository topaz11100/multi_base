import os
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_ROOT = os.path.abspath(os.path.join(THIS_DIR, "../../.."))
sys.path.insert(0, PROJ_ROOT)

from src.common.snn_builder import SNNConfig, build_snn_classifier  # noqa: E402


def build(
    model_name: str,
    input_dim: int,
    num_classes: int,
    hidden_dim: int,
    branch: int,
    S_min: float,
    S_max: float,
    th_len: int,
):
    cfg = SNNConfig(
        model_name=model_name,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        branch=branch,
        S_min=S_min,
        S_max=S_max,
        th_len=th_len,
    )
    return build_snn_classifier(cfg)
