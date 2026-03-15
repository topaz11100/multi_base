from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from src.common.surrogate import SpikeFn
from src.common.readout import apply_readout


@dataclass
class SNNConfig:
    model_name: str
    input_dim: int
    hidden_dim: int
    num_classes: int

    # ------------------------------------------------------------------
    # Dendrite / structure config
    # ------------------------------------------------------------------
    # Maximum number of dendritic branches (tensor shape).
    # Used by dendritic/resonant models (DH-SNN, D-RF, proposed variants).
    branch: int = 8

    # Continuous structural parameter s constraint for proposed variable-branch models.
    # - s is continuous
    # - D_int = floor(s)
    # - active branches are 1..D_int
    # Requirement: d_min/d_max 제거 -> (S_min, S_max)만 제공
    S_min: float = 1.0
    S_max: Optional[float] = None

    # D-RF adaptive threshold kernel length
    th_len: int = 4

    # Threshold/reset init (PLIF always learns both)
    v_th: float = 1.0
    v_reset: float = 1.0

    # D-RF pre-threshold scaling (adaptive threshold baseline)
    v_pre: float = 1.0

    # Surrogate spike
    spike_surrogate: str = "mg"


class SNNClassifier(nn.Module):
    """2-layer SNN classifier.

    Readout is fixed to **membrane potential** (mean output-layer `soma_state`).

      x -> layer1 -> layer2 -> mean_t(soma_state) -> logits

    Each layer is a dense spiking layer from src/neurons.
    """

    def __init__(self, layer1: nn.Module, layer2: nn.Module):
        super().__init__()
        self.layer1 = layer1
        self.layer2 = layer2

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        y1 = self.layer1.forward_sequence(x_seq, record=False)  # (B,T,H)
        _, rec2 = self.layer2.forward_sequence(y1, record=("soma_state",))  # (B,T,C)
        return apply_readout(soma_seq=rec2.get("soma_state"))

    def forward_with_records(self, x_seq: torch.Tensor):
        y1, rec1 = self.layer1.forward_sequence(x_seq, record=True)
        _, rec2 = self.layer2.forward_sequence(y1, record=True)
        logits = apply_readout(soma_seq=rec2.get("soma_state"))
        return logits, [rec1, rec2]

    def regularization_loss(self, lambda_ortho: float = 0.0, lambda_s: float = 0.0) -> torch.Tensor:
        # NOTE: s-complexity is defined as a *global* mean over all neurons in the model:
        #   L_s = (1/N_total) * sum_over_all_neurons s
        # Therefore, we compute s-regularization at the model level (not as a sum of per-layer means).
        from src.common.model_utils import s_complexity_mean

        loss = None
        for layer in (self.layer1, self.layer2):
            if hasattr(layer, "regularization_loss") and callable(getattr(layer, "regularization_loss")):
                # Avoid double-counting s: each layer may implement its own lambda_s term.
                l = layer.regularization_loss(lambda_ortho=lambda_ortho, lambda_s=0.0)  # type: ignore
                loss = l if loss is None else (loss + l)

        if loss is None:
            loss = torch.zeros((), device=next(self.parameters()).device)

        if lambda_s != 0.0:
            loss = loss + float(lambda_s) * s_complexity_mean(self)

        return loss


def build_layer(
    model_name: str,
    input_dim: int,
    output_dim: int,
    cfg: SNNConfig,
) -> nn.Module:
    name = model_name.lower()
    spike_fn = SpikeFn(name=cfg.spike_surrogate, lens=0.5, gamma=0.5)

    # ------------------------------------------------------------------
    # Baselines
    # ------------------------------------------------------------------
    if name in ("lif",):
        from src.neurons.LIF_neuron import LIFDenseLayer

        return LIFDenseLayer(input_dim, output_dim, alpha=0.9, v_th=cfg.v_th, spike_fn=spike_fn)

    if name in ("plif",):
        from src.neurons.PLIF_neuron import PLIFDenseLayer

        # Requirement: threshold/reset are ALWAYS learnable (no fixed 옵션)
        return PLIFDenseLayer(
            input_dim,
            output_dim,
            v_th=cfg.v_th,
            v_reset=cfg.v_reset,
            spike_fn=spike_fn,
        )

    if name in ("tc-lif", "tclif"):
        from src.neurons.TC_LIF_neuron import TCLIFDenseLayer

        return TCLIFDenseLayer(input_dim, output_dim, v_th=cfg.v_th, spike_fn=spike_fn)

    if name in ("ts-lif", "tslif"):
        from src.neurons.TS_LIF_neuron import TSLIFDenseLayer

        return TSLIFDenseLayer(input_dim, output_dim, v_th=cfg.v_th, spike_fn=spike_fn)

    if name in ("dh-snn", "dhsnn"):
        from src.neurons.DH_SNN_neuron import DHSNNDenseLayer

        return DHSNNDenseLayer(input_dim, output_dim, branch=cfg.branch, v_th=cfg.v_th, spike_fn=spike_fn)

    if name in ("d-rf", "drf"):
        from src.neurons.D_RF_neuron import DRFDenseLayer

        return DRFDenseLayer(
            input_dim,
            output_dim,
            branch=cfg.branch,
            th_len=cfg.th_len,
            v_pre=cfg.v_pre,
            spike_fn=spike_fn,
        )

    # ------------------------------------------------------------------
    # Proposed (my-*) variants
    # ------------------------------------------------------------------
    if name in ("my-lif", "mylif"):
        from src.neurons.my_LIF_neuron import MyLIFDenseLayer

        return MyLIFDenseLayer(input_dim, output_dim, v_th=cfg.v_th, spike_fn=spike_fn)

    if name in ("my-dh-snn", "my_dh-snn", "my_dh_snn", "my-dh"):
        from src.neurons.my_DH_SNN_neuron import MyDHSNNDenseLayer

        return MyDHSNNDenseLayer(
            input_dim,
            output_dim,
            branch=cfg.branch,
            S_min=cfg.S_min,
            S_max=cfg.S_max,
            v_th=cfg.v_th,
            spike_fn=spike_fn,
        )

    if name in ("my-r-dh-snn", "my-r-snn", "my_r-dh-snn", "my_r-snn", "my_r_dh_snn"):
        from src.neurons.my_R_DH_SNN_neuron import MyReverseDHSNNDenseLayer

        return MyReverseDHSNNDenseLayer(
            input_dim,
            output_dim,
            branch=cfg.branch,
            S_min=cfg.S_min,
            S_max=cfg.S_max,
            v_th=cfg.v_th,
            spike_fn=spike_fn,
        )

    if name in ("my-d-rf", "my_d-rf", "my_d_rf", "my-drf"):
        from src.neurons.my_D_RF_neuron import MyDRFDenseLayer

        return MyDRFDenseLayer(
            input_dim,
            output_dim,
            branch=cfg.branch,
            S_min=cfg.S_min,
            S_max=cfg.S_max,
            th_len=cfg.th_len,
            v_pre=cfg.v_pre,
            spike_fn=spike_fn,
        )

    raise KeyError(f"Unknown model_name: {model_name}")



def _disable_output_spikes_(layer: nn.Module, *, v_th: float = 1e9) -> None:
    """Disable spiking/reset in the output layer for membrane-potential readout.

    The classifier readout is fixed to:

        logits = mean_t( soma_state[t] )

    For LIF/DH-style layers, spiking induces a soft reset (mem <- mem - v_th * spk),
    which can undesirably clamp the output-layer membrane dynamics. Setting a very
    large threshold makes spikes never occur, turning the output layer into a pure
    leaky integrator for stable logits.

    For PLIF, threshold/reset are learnable via (v_th_raw, v_reset_raw); we set those
    raw parameters to a large value.
    """
    # Fixed-threshold layers (LIF, TC-LIF, TS-LIF, DH-SNN, my-*)
    if hasattr(layer, "v_th") and isinstance(getattr(layer, "v_th"), (float, int)):
        setattr(layer, "v_th", float(v_th))
        return

    # Learnable-threshold layers (PLIF)
    if hasattr(layer, "v_th_raw"):
        with torch.no_grad():
            getattr(layer, "v_th_raw").fill_(float(v_th))
            if hasattr(layer, "v_reset_raw"):
                getattr(layer, "v_reset_raw").fill_(float(v_th))

def build_snn_classifier(cfg: SNNConfig) -> SNNClassifier:
    l1 = build_layer(cfg.model_name, cfg.input_dim, cfg.hidden_dim, cfg)
    l2 = build_layer(cfg.model_name, cfg.hidden_dim, cfg.num_classes, cfg)

    # Output layer is used with membrane-potential readout (mean over time of soma_state).
    # Disable spiking/reset in the output layer to avoid clamping logits.
    _disable_output_spikes_(l2)

    return SNNClassifier(l1, l2)
