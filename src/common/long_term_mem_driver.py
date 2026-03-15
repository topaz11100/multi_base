from __future__ import annotations

import os
from dataclasses import asdict
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm.auto import tqdm

from src.common.model_utils import (
    aggregate_breakdowns,
    format_breakdown_table,
    layer_active_param_breakdown,
)
from src.common.long_term_mem_dataset import ensure_serial_xor_datasets
from src.common.plotting import save_hist_line, save_line_plot
from src.common.snn_builder import SNNConfig, build_layer, _disable_output_spikes_
from src.common.utils import ensure_dir, get_device, now_timestamp_seoul, save_text, set_seed, derive_branch_from_S_max

# Keep consistent with accuracy benchmark
ALL_MODELS = [
    "lif",
    "plif",
    "tc-lif",
    "ts-lif",
    "dh-snn",
    "d-rf",
    "my-lif",
    "my-dh-snn",
    "my-r-dh-snn",
    "my-d-rf",
]


# -----------------------------------------------------------------------------
# Synthetic XOR tasks (adapted from Origin/DH-SNN-main)
# -----------------------------------------------------------------------------


def _rates_tensor(low: float, high: float, device: torch.device) -> torch.Tensor:
    return torch.tensor([float(low), float(high)], device=device, dtype=torch.float32)


@torch.no_grad()
def generate_delayed_xor_batch(
    batch_size: int,
    time_steps: int,
    channel_size: int,
    coding_time: int,
    test_time: int,
    noise_rate: float,
    rate_low: float,
    rate_high: float,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate one batch for delayed XOR.

    Returns:
      x: (B,T,C) float in {0,1}
      y: (B,T) float in {0,1} (label broadcast)
      mask: (B,T) bool (valid timesteps for loss/acc)

    Mask matches DH-SNN code: valid when t > start_of_second_signal.
    """
    B = int(batch_size)
    T = int(time_steps)
    C = int(channel_size)
    K = int(coding_time)
    test_time = int(test_time)

    if K <= 0 or K >= T:
        raise ValueError(f"coding_time must be in (0, time_steps), got {K} vs {T}")
    if test_time <= 0:
        raise ValueError(f"test_time must be >=1, got {test_time}")

    rates = _rates_tensor(rate_low, rate_high, device)

    # base noise
    x = (torch.rand(B, T, C, device=device) < float(noise_rate))

    # first pattern at [0:K]
    init_pattern = torch.randint(0, 2, (B,), device=device)
    p1 = rates[init_pattern].view(B, 1, 1)
    x[:, :K, :] |= (torch.rand(B, K, C, device=device) < p1)

    # second pattern near end, shifted by position in {0,...,test_time-1}
    position = torch.randint(0, test_time, (B,), device=device)
    pattern = torch.randint(0, 2, (B,), device=device)
    label = (init_pattern != pattern).to(torch.float32)  # XOR

    p2 = rates[pattern].view(B, 1, 1)
    add2 = (torch.rand(B, K, C, device=device) < p2)

    start = T - (position + 1) * K  # (B,)
    # advanced indexing assignment per sample
    batch_idx = torch.arange(B, device=device).view(B, 1)
    time_idx = start.view(B, 1) + torch.arange(K, device=device).view(1, K)
    x[batch_idx, time_idx, :] |= add2

    t = torch.arange(T, device=device).view(1, T)
    mask = t > start.view(B, 1)

    y = label.view(B, 1).expand(B, T)
    return x.to(torch.float32), y, mask


@torch.no_grad()
def generate_multiscale_xor_batch(
    batch_size: int,
    time_steps: int,
    channel_size: int,
    coding_time: int,
    remain_time: int,
    start_time: int,
    noise_rate: float,
    rate_low: float,
    rate_high: float,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate one batch for multitimescale XOR.

    Returns:
      x: (B,T,2*C) float in {0,1}
      y: (B,T) float in {0,1} (piecewise-constant per segment)
      mask: (B,T) bool (valid timesteps for loss/acc)

    Mask matches DH-SNN code:
      mask[t] = (t > start_time) and ((t-start_time) % (coding+remain) > remain_time)
    """
    B = int(batch_size)
    T = int(time_steps)
    C = int(channel_size)
    K = int(coding_time)
    R = int(remain_time)
    S = int(start_time)

    if S <= 0 or S >= T:
        raise ValueError(f"start_time must be in (0, time_steps), got {S} vs {T}")
    if K <= 0:
        raise ValueError(f"coding_time must be >=1, got {K}")
    if R < 0:
        raise ValueError(f"remain_time must be >=0, got {R}")
    L = K + R
    if L <= 1:
        raise ValueError(f"coding_time+remain_time must be >=2, got {L}")
    num_segments = (T - S) // L
    if num_segments <= 0:
        raise ValueError(
            f"time_steps-start_time must be >= (coding_time+remain_time), got (T={T}, S={S}, L={L})"
        )

    rates = _rates_tensor(rate_low, rate_high, device)

    x = (torch.rand(B, T, 2 * C, device=device) < float(noise_rate))

    init_pattern = torch.randint(0, 2, (B,), device=device)
    p1 = rates[init_pattern].view(B, 1, 1)
    x[:, :S, :C] |= (torch.rand(B, S, C, device=device) < p1)

    y = torch.zeros(B, T, device=device, dtype=torch.float32)

    for seg in range(num_segments):
        window_start = S + seg * L
        window_end = S + (seg + 1) * L

        pattern = torch.randint(0, 2, (B,), device=device)
        label = (init_pattern != pattern).to(torch.float32)

        # label for whole window
        y[:, window_start:window_end] = label.view(B, 1)

        # spikes for second channel in coding part (starts after remain_time)
        code_start = window_start + R
        code_end = code_start + K
        if code_end > T:
            break
        p2 = rates[pattern].view(B, 1, 1)
        x[:, code_start:code_end, C:] |= (torch.rand(B, K, C, device=device) < p2)

    t = torch.arange(T, device=device)
    mask = (t > S) & (((t - S) % L) > R)
    mask = mask.view(1, T).expand(B, T)

    return x.to(torch.float32), y, mask


# -----------------------------------------------------------------------------
# Single-neuron model wrapper (logits = soma_state sequence)
# -----------------------------------------------------------------------------


class SequenceBinaryXOR(nn.Module):
    def __init__(self, layers: List[nn.Module]):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        h = x_seq
        for li, layer in enumerate(self.layers):
            is_last = (li == len(self.layers) - 1)
            if is_last:
                _, rec = layer.forward_sequence(h, record=("soma_state",))
                return rec["soma_state"].squeeze(-1)
            h = layer.forward_sequence(h, record=False)
        raise RuntimeError("empty model")

    def regularization_loss(self, lambda_ortho: float = 0.0, lambda_s: float = 0.0) -> torch.Tensor:
        # NOTE: s-complexity is defined as a *global* mean over all neurons in the model:
        #   L_s = (1/N_total) * sum_over_all_neurons s
        # Therefore, we compute s-regularization at the model level.
        from src.common.model_utils import s_complexity_mean

        loss = None
        for layer in self.layers:
            if hasattr(layer, "regularization_loss"):
                l = layer.regularization_loss(lambda_ortho=lambda_ortho, lambda_s=0.0)  # type: ignore
                loss = l if loss is None else (loss + l)
        if loss is None:
            loss = torch.zeros((), device=next(self.parameters()).device)

        if lambda_s != 0.0:
            loss = loss + float(lambda_s) * s_complexity_mean(self)

        return loss


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------


def _masked_bce_loss(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # logits: (B,T), targets: (B,T), mask: (B,T)
    logits_m = logits[mask]
    targets_m = targets[mask]
    return F.binary_cross_entropy_with_logits(logits_m, targets_m)


@torch.no_grad()
def _masked_bce_sum_and_correct(
    logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor
) -> Tuple[float, int, int]:
    logits_m = logits[mask]
    targets_m = targets[mask]
    loss_sum = F.binary_cross_entropy_with_logits(logits_m, targets_m, reduction="sum").item()
    pred = (torch.sigmoid(logits_m) > 0.5).to(targets_m.dtype)
    correct = int((pred == targets_m).sum().item())
    count = int(targets_m.numel())
    return float(loss_sum), correct, count


def _save_single_model_distributions(layer: nn.Module, model_dir: str) -> None:
    """Save timing / structure distributions (line histograms)."""
    timing = {}
    if hasattr(layer, "get_timing_params"):
        timing = layer.get_timing_params()  # type: ignore
    struct = {}
    if hasattr(layer, "get_structure_params"):
        struct = layer.get_structure_params()  # type: ignore

    # Optional active-only mask for branch-wise timing params (alpha/tau/omega)
    mask = None
    N = None
    D = None

    # timing
    if timing:
        tdir = ensure_dir(os.path.join(model_dir, "timing"))

        # Optional active-only mask for branch-wise timing params (alpha/tau/omega)
        mask = None
        N = None
        D = None
        if hasattr(layer, "soft_mask") and callable(getattr(layer, "soft_mask")):
            try:
                m = layer.soft_mask(torch.float32)  # type: ignore
                if torch.is_tensor(m):
                    mask = m.detach().cpu().numpy()
                    N = int(getattr(layer, "output_dim"))
                    D = int(getattr(layer, "branch"))
            except Exception:
                mask = None
                N = None
                D = None

        for k, v in timing.items():
            try:
                if torch.is_tensor(v):
                    arr = v.detach().cpu().flatten().numpy()
                else:
                    arr = np.asarray(v, dtype=float).reshape(-1)

                if k in ("alpha", "tau", "omega") and mask is not None and N is not None and D is not None:
                    try:
                        if mask.shape == (N, D) and arr.size == (N * D):
                            arr2 = arr.reshape(N, D)
                            arr = arr2[mask > 0.0].reshape(-1)
                    except Exception:
                        pass

                save_hist_line(os.path.join(tdir, f"layer1_{k}.png"), arr, xlabel=k)
            except Exception:
                # Distribution plots must never crash the experiment.
                continue


    # structure (e.g., s)
    if struct:
        sdir = ensure_dir(os.path.join(model_dir, "structure"))
        for k, v in struct.items():
            try:
                if torch.is_tensor(v):
                    arr = v.detach().cpu().flatten().numpy()
                else:
                    arr = np.asarray(v, dtype=float).reshape(-1)
                save_hist_line(os.path.join(sdir, f"layer1_{k}.png"), arr, xlabel=k)
            except Exception:
                continue

    # model-level (single layer == model)
    if timing:
        for k, v in timing.items():
            try:
                if torch.is_tensor(v):
                    arr = v.detach().cpu().flatten().numpy()
                else:
                    arr = np.asarray(v, dtype=float).reshape(-1)

                if k in ("alpha", "tau", "omega") and mask is not None and N is not None and D is not None:
                    try:
                        if mask.shape == (N, D) and arr.size == (N * D):
                            arr2 = arr.reshape(N, D)
                            arr = arr2[mask > 0.0].reshape(-1)
                    except Exception:
                        pass

                save_hist_line(os.path.join(tdir, f"model_{k}.png"), arr, xlabel=k)
            except Exception:
                continue
    if struct:
        for k, v in struct.items():
            try:
                if torch.is_tensor(v):
                    arr = v.detach().cpu().flatten().numpy()
                else:
                    arr = np.asarray(v, dtype=float).reshape(-1)
                save_hist_line(os.path.join(sdir, f"model_{k}.png"), arr, xlabel=k)
            except Exception:
                continue


@torch.no_grad()
def _evaluate_xor(
    net: SequenceBinaryXOR,
    x: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor,
    device: torch.device,
    eval_batch_size: int,
) -> Tuple[float, float]:
    """Return (loss_mean, acc_mean) over masked timesteps."""
    net.eval()
    loss_sum = 0.0
    correct = 0
    count = 0
    N = int(x.shape[0])
    bs = max(1, int(eval_batch_size))
    for st in range(0, N, bs):
        xb = x[st:st + bs].to(device)
        yb = y[st:st + bs].to(device)
        mb = mask[st:st + bs].to(device)
        logits = net(xb)
        ls, c, n = _masked_bce_sum_and_correct(logits, yb, mb)
        loss_sum += ls
        correct += c
        count += n

    loss_mean = loss_sum / max(1, count)
    acc_mean = float(correct) / max(1, count)
    return float(loss_mean), float(acc_mean)


def _build_sequence_xor_net(
    model_name: str,
    input_dim: int,
    hidden_dims: Sequence[int],
    branch: int,
    S_min: float,
    S_max: Optional[float],
    th_len: int,
    v_th: float,
    v_reset: float,
    v_pre: float,
) -> SequenceBinaryXOR:
    cfg = SNNConfig(
        model_name=model_name,
        input_dim=input_dim,
        hidden_dim=1,
        num_classes=1,
        branch=int(branch),
        S_min=float(S_min),
        S_max=None if S_max is None else float(S_max),
        th_len=th_len,
        v_th=v_th,
        v_reset=v_reset,
        v_pre=v_pre,
    )
    dims = [input_dim] + list(hidden_dims) + [1]
    layers: List[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(build_layer(model_name, dims[i], dims[i + 1], cfg))
    _disable_output_spikes_(layers[-1])
    return SequenceBinaryXOR(layers)


# -----------------------------------------------------------------------------
# Main driver
# -----------------------------------------------------------------------------


def run_long_term_mem_xor(
    task: str,
    models: Sequence[str],
    out_root: str,
    data_root: str,
    seed: int,
    device: str = "auto",
    # optimizer / schedule
    epochs: int = 50,
    soft_mask_epochs: Optional[int] = None,
    stabilize_epochs: int = 0,
    ste_epochs: int = 0,
    steps_per_epoch: int = 100,
    batch_size: int = 500,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    weight_decay_dend_soma: Optional[float] = None,
    check_every: int = 1,
    eval_batches: int = 20,
    hidden: Sequence[int] = (256,),
    # neuron structure
    S_min: float = 1.0,
    S_max: float = 8.0,
    th_len: int = 4,
    v_th: float = 1.0,
    v_reset: Optional[float] = None,
    v_pre: float = 1.0,
    lambda_ortho: float = 0.0,
    lambda_s: float = 0.0,
    # delayed XOR params
    delayed_time_steps: int = 200,
    delayed_channel_size: int = 20,
    delayed_coding_time: int = 10,
    delayed_test_time: int = 1,
    # multiscale XOR params
    multi_time_steps: int = 100,
    multi_channel_size: int = 20,
    multi_coding_time: int = 10,
    multi_remain_time: int = 5,
    multi_start_time: int = 10,
    # shared
    noise_rate: float = 0.01,
    rate_low: float = 0.2,
    rate_high: float = 0.6,
    exp_name: Optional[str] = None,
    timestamp: Optional[str] = None,
) -> str:
    """Run long-term memory XOR experiments with fixed serial datasets."""
    if v_reset is None:
        v_reset = float(v_th)

    set_seed(seed)
    dev = get_device(device)

    S_min = float(S_min)
    S_max = float(S_max)
    if not (S_max > 0.0):
        raise ValueError(f"S_max must be > 0 (got S_max={S_max})")
    if not (1.0 <= S_min <= S_max):
        raise ValueError(f"Require 1 <= S_min <= S_max (got S_min={S_min}, S_max={S_max})")

    branch = int(derive_branch_from_S_max(S_max))

    ts = str(timestamp) if timestamp is not None else now_timestamp_seoul()
    default_exp_name = f"basic_long_term_mem-{task}"
    exp_name_final = (exp_name or default_exp_name).replace(" ", "").replace("/", "-")
    out_dir = ensure_dir(os.path.join(out_root, f"{exp_name_final}_{ts}"))

    # hyperparams.txt (not JSON)
    hp_lines: List[str] = []
    hp_lines.append(f"experiment={exp_name_final}")
    hp_lines.append(f"timestamp={ts}")
    hp_lines.append(f"task={task}")
    hp_lines.append(f"device={dev}")
    hp_lines.append(f"seed={seed}")
    hp_lines.append("")
    soft_e = int(epochs) if soft_mask_epochs is None else int(soft_mask_epochs)
    stb_e = int(stabilize_epochs)
    ste_e = int(ste_epochs)
    if ste_e < 0:
        raise ValueError(f"ste_epochs must be >= 0 (got ste_epochs={ste_e})")
    if int(stb_e) <= 0:
        ste_e = 0
    if ste_e > int(soft_e):
        ste_e = int(soft_e)
    total_e = int(soft_e + stb_e)


    hp_lines.append(f"epochs={total_e}")
    hp_lines.append(f"soft_mask_epochs={soft_e}")
    hp_lines.append(f"stabilize_epochs={stb_e}")
    hp_lines.append(f"ste_epochs={ste_e}")
    hp_lines.append(f"steps_per_epoch={steps_per_epoch}")
    hp_lines.append(f"batch_size={batch_size}")
    hp_lines.append(f"lr={lr}")
    hp_lines.append(f"weight_decay={weight_decay} (AdamW; layer connection weights only)")
    hp_lines.append(
        f"weight_decay_dend_soma={weight_decay_dend_soma} (my_R_DH_SNN W_mix only; None => follow weight_decay)"
    )
    hp_lines.append(f"check_every={check_every}")
    hp_lines.append(f"eval_batches={eval_batches}")
    hp_lines.append(f"hidden={list(hidden)}")
    hp_lines.append("")
    hp_lines.append(f"S_max={S_max} -> branch={branch} (derived)")
    hp_lines.append(f"S_min={S_min}")
    hp_lines.append(f"S_max={S_max}")
    hp_lines.append(f"th_len={th_len}")
    hp_lines.append(f"v_th={v_th}")
    hp_lines.append(f"v_reset={v_reset}")
    hp_lines.append(f"PLIF: v_th/v_reset are always learnable (init only)")
    hp_lines.append(f"v_pre={v_pre}")
    hp_lines.append(f"lambda_ortho={lambda_ortho}")
    hp_lines.append(f"lambda_s={lambda_s}")
    hp_lines.append("")
    hp_lines.append(f"noise_rate={noise_rate}")
    hp_lines.append(f"rate_low={rate_low}")
    hp_lines.append(f"rate_high={rate_high}")
    hp_lines.append("")
    if task == "delayed_XOR":
        hp_lines.append(f"time_steps={delayed_time_steps}")
        hp_lines.append(f"channel_size={delayed_channel_size}")
        hp_lines.append(f"coding_time={delayed_coding_time}")
        hp_lines.append(f"test_time={delayed_test_time}")
    elif task == "multiscale_XOR":
        hp_lines.append(f"time_steps={multi_time_steps}")
        hp_lines.append(f"channel_size={multi_channel_size}")
        hp_lines.append(f"coding_time={multi_coding_time}")
        hp_lines.append(f"remain_time={multi_remain_time}")
        hp_lines.append(f"start_time={multi_start_time}")
    else:
        raise ValueError(f"Unknown task: {task}")

    save_text(os.path.join(out_dir, "hyperparams.txt"), "\n".join(hp_lines) + "\n")

    ds_paths = ensure_serial_xor_datasets(
        data_root_abs=data_root,
        seed=int(seed),
        p_low=float(rate_low),
        p_high=float(rate_high),
        p_noise=float(noise_rate),
        Ls=int(delayed_coding_time if task == "delayed_XOR" else multi_coding_time),
        Ld=max(1, int(delayed_time_steps - 2 * delayed_coding_time)),
        Lg=int(multi_remain_time),
        K=max(1, int((multi_time_steps - multi_start_time) // max(1, (multi_coding_time + multi_remain_time)))),
    )

    # model structure + active parameter breakdown
    param_report_lines: List[str] = []
    param_report_lines.append(f"experiment: {exp_name_final}")
    param_report_lines.append(f"timestamp: {ts}")
    param_report_lines.append(f"task: {task}")
    param_report_lines.append(f"seed: {seed}")
    param_report_lines.append(f"device: {dev}")
    param_report_lines.append("")

    results_lines: List[str] = []
    results_lines.append(f"experiment: {exp_name_final}")
    results_lines.append(f"timestamp: {ts}")
    results_lines.append(f"task: {task}")
    results_lines.append("")

    def _load_delayed(split: str):
        z = np.load(ds_paths[f"delayed_{split}"])
        x = torch.from_numpy(z["x"].astype(np.float32)).repeat(1, 1, int(delayed_channel_size))
        y = torch.zeros((x.shape[0], x.shape[1]), dtype=torch.float32)
        m = torch.zeros((x.shape[0], x.shape[1]), dtype=torch.bool)
        idx = torch.from_numpy(z["eval_idx"].astype(np.int64))
        lbl = torch.from_numpy(z["y"].astype(np.float32))
        y[torch.arange(x.shape[0]), idx] = lbl
        m[torch.arange(x.shape[0]), idx] = True
        return x, y, m

    def _load_multi(split: str):
        z = np.load(ds_paths[f"multi_{split}"])
        base = torch.from_numpy(z["x"].astype(np.float32)).repeat(1, 1, int(multi_channel_size))
        x = torch.cat([base, base], dim=2)
        y = torch.zeros((x.shape[0], x.shape[1]), dtype=torch.float32)
        m = torch.zeros((x.shape[0], x.shape[1]), dtype=torch.bool)
        qidx = torch.from_numpy(z["query_eval_idx"].astype(np.int64))
        yseq = torch.from_numpy(z["y_seq"].astype(np.float32))
        bi = torch.arange(x.shape[0]).unsqueeze(1).expand_as(qidx)
        y[bi, qidx] = yseq
        m[bi, qidx] = True
        return x, y, m

    if task == "delayed_XOR":
        input_dim = int(delayed_channel_size)
        train_x, train_y, train_m = _load_delayed("train")
        test_x, test_y, test_m = _load_delayed("test")
    else:
        input_dim = int(multi_channel_size) * 2
        train_x, train_y, train_m = _load_multi("train")
        test_x, test_y, test_m = _load_multi("test")

    for mname in models:
        model_dir = ensure_dir(os.path.join(out_dir, mname))
        S_max_eff = float(S_max)

        net = _build_sequence_xor_net(
            model_name=mname,
            input_dim=input_dim,
            hidden_dims=hidden,
            branch=branch,
            S_min=S_min,
            S_max=S_max_eff,
            th_len=th_len,
            v_th=v_th,
            v_reset=float(v_reset),
            v_pre=v_pre,
        ).to(dev)

        # param breakdown
        param_report_lines.append(f"[{mname}] model_structure")
        param_report_lines.append(f"  {input_dim} -> 1")
        try:
            all_bd = [layer_active_param_breakdown(layer) for layer in net.layers]
            for li, bd_layer in enumerate(all_bd, start=1):
                param_report_lines.append(f"[{mname}] layer{li} initial_active_param_breakdown")
                param_report_lines.append(format_breakdown_table(bd_layer, prefix="  "))
            param_report_lines.append(f"[{mname}] initial_total_active_params = {sum(sum(b.values()) for b in all_bd)}")
        except Exception as e:
            param_report_lines.append(f"[{mname}] breakdown_error: {type(e).__name__}: {e}")
        param_report_lines.append("")

        from src.common.optim import build_adamw

        opt, _ = build_adamw(
            net,
            lr=float(lr),
            weight_decay=float(weight_decay),
            weight_decay_dend_soma=weight_decay_dend_soma,
        )

        train_epochs: List[int] = []
        train_accs: List[float] = []
        test_accs: List[float] = []
        train_losses: List[float] = []
        test_losses: List[float] = []

        best_test_acc = -1.0
        best_epoch = -1

        pbar = tqdm(range(1, int(total_e) + 1), desc=mname, total=int(total_e), leave=True)
        for epoch in pbar:
            # STE schedule (last `ste_e` epochs of stage A): forward hard / backward soft.
            if int(stb_e) > 0:
                from src.common.model_utils import set_ste_mode_
                ste_on = int(ste_e) > 0 and (int(epoch) >= int(soft_e) - int(ste_e) + 1) and (int(epoch) <= int(soft_e))
                set_ste_mode_(net, bool(ste_on))
            else:
                from src.common.model_utils import set_ste_mode_
                set_ste_mode_(net, False)

            if int(stb_e) > 0 and int(epoch) == int(soft_e) + 1:
                # Transition: soft mask -> hard mask and freeze s.
                from src.common.model_utils import harden_variable_branches_

                harden_variable_branches_(net)
            net.train()
            for _ in range(int(steps_per_epoch)):
                idx = torch.randint(0, train_x.shape[0], (int(batch_size),))
                x = train_x[idx].to(dev)
                y = train_y[idx].to(dev)
                mask = train_m[idx].to(dev)
                opt.zero_grad(set_to_none=True)
                logits = net(x)
                loss = _masked_bce_loss(logits, y, mask)
                loss = loss + net.regularization_loss(lambda_ortho=lambda_ortho, lambda_s=lambda_s)
                loss.backward()
                opt.step()

            if check_every > 0 and (epoch % int(check_every) == 0 or epoch == 1 or epoch == int(total_e)):
                tr_loss, tr_acc = _evaluate_xor(net, train_x, train_y, train_m, dev, eval_batch_size=int(batch_size))
                te_loss, te_acc = _evaluate_xor(net, test_x, test_y, test_m, dev, eval_batch_size=int(batch_size))

                train_epochs.append(epoch)
                train_accs.append(tr_acc)
                test_accs.append(te_acc)
                train_losses.append(tr_loss)
                test_losses.append(te_loss)

                if te_acc > best_test_acc:
                    best_test_acc = te_acc
                    best_epoch = epoch
                pbar.set_postfix(
                    train_acc=f"{tr_acc:.4f}",
                    test_acc=f"{te_acc:.4f}",
                    train_loss=f"{tr_loss:.4f}",
                    test_loss=f"{te_loss:.4f}",
                    best=f"{best_test_acc:.4f}@{best_epoch}",
                )

        # save curves
        if train_epochs:
            save_line_plot(
                os.path.join(model_dir, "acc_curve.png"),
                {"train": train_accs, "test": test_accs},
                x=train_epochs,
                xlabel="epoch",
                ylabel="accuracy",
                title=f"{mname} accuracy",
            )

        # final active parameter breakdown (post-training)
        try:
            all_bd_f = [layer_active_param_breakdown(layer) for layer in net.layers]
            for li, bd_layer_f in enumerate(all_bd_f, start=1):
                param_report_lines.append(f"[{mname}] layer{li} final_active_param_breakdown")
                param_report_lines.append(format_breakdown_table(bd_layer_f, prefix="  "))
            param_report_lines.append(f"[{mname}] final_total_active_params = {sum(sum(b.values()) for b in all_bd_f)}")
            # total params (trainable) is constant but record for completeness
            param_report_lines.append(f"[{mname}] total_params = {sum(int(p.numel()) for p in net.parameters())}")
        except Exception as e:
            param_report_lines.append(f"[{mname}] final_breakdown_error: {type(e).__name__}: {e}")
        param_report_lines.append("")

        # distributions
        for layer in net.layers:
            _save_single_model_distributions(layer, model_dir)

        # checkpoint
        torch.save({"model": net.state_dict(), "config": {"model": mname, "task": task}}, os.path.join(model_dir, "final.pt"))

        results_lines.append(f"[{mname}] best_test_acc={best_test_acc:.6f} @ epoch={best_epoch}")
        if train_epochs:
            results_lines.append(
                f"[{mname}] last: epoch={train_epochs[-1]} train_acc={train_accs[-1]:.6f} test_acc={test_accs[-1]:.6f}"
            )
        results_lines.append("")

    save_text(os.path.join(out_dir, "results.txt"), "\n".join(results_lines) + "\n")
    save_text(os.path.join(out_dir, "model_structure_and_params.txt"), "\n".join(param_report_lines) + "\n")
    return out_dir
