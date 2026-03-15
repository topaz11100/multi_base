from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from tqdm.auto import tqdm

from src.common.datasets import (
    get_smnist_loaders,
    get_scifar10_loaders,
    get_shd_loaders,
    get_ssc_loaders,
)
from src.common.model_utils import (
    count_active_parameters,
    count_parameters,
    layer_active_param_breakdown,
    aggregate_breakdowns,
    format_breakdown_table,
)
from src.common.plotting import save_hist_line, save_line_plot
from src.common.readout import apply_readout
from src.common.utils import ensure_dir, now_timestamp_seoul, save_text, set_seed, derive_branch_from_S_max, float_to_tag
from src.common.training import evaluate_classifier


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


def _load_dataset(
    dataset: str,
    data_root: str,
    batch_size: int,
    num_workers: int,
    download: bool,
    cifar_mode: str = "parallel",
    T_event: int = 250,
    seed: Optional[int] = None,
):
    d = dataset.lower()
    if d in ("s-mnist", "smnist"):
        train_loader, test_loader, num_classes, input_dim, T = get_smnist_loaders(
            data_root, batch_size=batch_size, num_workers=num_workers, download=download, seed=seed
        )
    elif d in ("s-cifar10", "scifar10"):
        train_loader, test_loader, num_classes, input_dim, T = get_scifar10_loaders(
            data_root, batch_size=batch_size, num_workers=num_workers, download=download, mode=cifar_mode, seed=seed
        )
    elif d == "shd":
        train_loader, test_loader, num_classes, input_dim, T = get_shd_loaders(
            data_root, batch_size=batch_size, num_workers=num_workers, download=download, T=T_event, seed=seed
        )
    elif d == "ssc":
        train_loader, test_loader, num_classes, input_dim, T = get_ssc_loaders(
            data_root, batch_size=batch_size, num_workers=num_workers, download=download, T=T_event, seed=seed
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    return train_loader, test_loader, num_classes, input_dim, T


def _build_mlp_snn(
    model_name: str,
    input_dim: int,
    hidden_dims: Sequence[int],
    num_classes: int,
    branch: int,
    S_min: float,
    S_max: Optional[float],
    th_len: int,
    v_th: float = 1.0,
    v_reset: Optional[float] = None,
    v_pre: float = 1.0,
) -> nn.Module:
    """
    Build a feed-forward SNN with hidden layers.
    All layers use the same neuron model_name (baseline or proposed).
    """
    from src.common.snn_builder import SNNConfig, build_layer, _disable_output_spikes_

    # Build sequential layers as ModuleList and a wrapper forward
    class FFNet(nn.Module):
        def __init__(self, layers: List[nn.Module]):
            super().__init__()
            self.layers = nn.ModuleList(layers)

        def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
            h = x_seq
            for li, layer in enumerate(self.layers):
                is_last = (li == len(self.layers) - 1)
                if is_last:
                    _, rec = layer.forward_sequence(h, record=True)
                    return apply_readout(soma_seq=rec.get("soma_state"))
                h = layer.forward_sequence(h, record=False)
            raise RuntimeError("FFNet requires at least one layer")

        def regularization_loss(self, lambda_ortho: float = 0.0, lambda_s: float = 0.0) -> torch.Tensor:
            # NOTE: s-complexity is defined as a *global* mean over all neurons in the model:
            #   L_s = (1/N_total) * sum_over_all_neurons s
            # Therefore, we compute s-regularization at the model level (not as a sum of per-layer means).
            from src.common.model_utils import s_complexity_mean

            loss = None
            for layer in self.layers:
                if hasattr(layer, "regularization_loss") and callable(getattr(layer, "regularization_loss")):
                    # Avoid double-counting s: each layer may implement its own lambda_s term.
                    l = layer.regularization_loss(lambda_ortho=lambda_ortho, lambda_s=0.0)  # type: ignore
                    loss = l if loss is None else (loss + l)
            if loss is None:
                loss = torch.zeros((), device=next(self.parameters()).device)

            if lambda_s != 0.0:
                loss = loss + float(lambda_s) * s_complexity_mean(self)

            return loss

    # SNNConfig is used only for passing shared params
    cfg = SNNConfig(
        model_name=model_name,
        input_dim=input_dim,
        hidden_dim=int(hidden_dims[0]) if len(hidden_dims) > 0 else num_classes,
        num_classes=num_classes,
        branch=int(branch),
        S_min=float(S_min),
        S_max=None if S_max is None else float(S_max),
        th_len=int(th_len),
        v_th=float(v_th),
        v_reset=float(v_th) if v_reset is None else float(v_reset),
        v_pre=float(v_pre),
        spike_surrogate="mg",
    )

    dims = [input_dim] + list(hidden_dims) + [num_classes]
    layers = []
    for i in range(len(dims) - 1):
        layers.append(build_layer(model_name, dims[i], dims[i + 1], cfg))

    # Output layer is used with membrane-potential readout (mean over time of soma_state).
    # Disable spiking/reset in the output layer to avoid clamping logits.
    _disable_output_spikes_(layers[-1])
    return FFNet(layers)


def _extract_timing(layer: nn.Module) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}

    # Optional active-only mask for branch-wise timing params (alpha/tau/omega)
    mask = None
    N = None
    D = None
    if hasattr(layer, 'soft_mask') and callable(getattr(layer, 'soft_mask')):
        try:
            m = layer.soft_mask(torch.float32)  # type: ignore
            if torch.is_tensor(m):
                mask = m.detach().cpu().numpy()
                N = int(getattr(layer, 'output_dim'))
                D = int(getattr(layer, 'branch'))
        except Exception:
            mask = None
            N = None
            D = None

    if hasattr(layer, 'get_timing_params') and callable(getattr(layer, 'get_timing_params')):
        tp = layer.get_timing_params()  # type: ignore
        for k, v in tp.items():
            if torch.is_tensor(v):
                arr = v.detach().cpu().numpy().reshape(-1)
            else:
                arr = np.asarray(v, dtype=float).reshape(-1)

            # Active-only for branch-wise timing params on variable-branch models
            if k in ('alpha', 'tau', 'omega') and mask is not None and N is not None and D is not None:
                try:
                    if mask.shape == (N, D) and arr.size == (N * D):
                        arr2 = arr.reshape(N, D)
                        arr = arr2[mask > 0.0].reshape(-1)
                except Exception:
                    pass

            out[k] = arr

    return out



def _extract_structure(layer: nn.Module) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    if hasattr(layer, "get_structure_params") and callable(getattr(layer, "get_structure_params")):
        sp = layer.get_structure_params()  # type: ignore
        for k, v in sp.items():
            if torch.is_tensor(v):
                out[k] = v.detach().cpu().numpy().reshape(-1)
            else:
                out[k] = np.asarray(v).reshape(-1)
    return out


def _save_model_distributions(model_dir: str, model: nn.Module, model_name: str) -> None:
    """
    Save final timing distributions at the end of training.
    """
    timing_dir = ensure_dir(os.path.join(model_dir, "timing"))
    # NOTE: Weight distribution plots are intentionally omitted here.
    # (freq_analysis has a dedicated active-only weight visualization; for the
    # acc_benchmark spec we only need timing/structure distributions.)

    # Collect per-layer and aggregated
    agg: Dict[str, List[np.ndarray]] = {}
    agg_s: List[np.ndarray] = []
    agg_D: List[np.ndarray] = []

    for li, layer in enumerate(model.layers):  # type: ignore
        lname = f"layer{li+1}"
        tp = _extract_timing(layer)
        sp = _extract_structure(layer)

        if "s" in sp:
            save_hist_line(os.path.join(timing_dir, f"{lname}_s.png"), sp["s"].reshape(-1), xlabel="s")
            agg_s.append(sp["s"].reshape(-1))
        if "D_int" in sp:
            save_hist_line(os.path.join(timing_dir, f"{lname}_D_int.png"), sp["D_int"].reshape(-1), xlabel="D_int")
            agg_D.append(sp["D_int"].reshape(-1))

        for k, v in tp.items():
            save_hist_line(os.path.join(timing_dir, f"{lname}_{k}.png"), v, xlabel=k)
            agg.setdefault(k, []).append(v)

    # Model-level
    for k, vs in agg.items():
        save_hist_line(os.path.join(timing_dir, f"model_{k}.png"), np.concatenate(vs), xlabel=k)

    if len(agg_s) > 0:
        save_hist_line(os.path.join(timing_dir, "model_s.png"), np.concatenate(agg_s, axis=0), xlabel="s")
    if len(agg_D) > 0:
        save_hist_line(os.path.join(timing_dir, "model_D_int.png"), np.concatenate(agg_D, axis=0), xlabel="D_int")

    # (no weights)


def _params_breakdown(model: nn.Module) -> Dict[str, Any]:
    """
    Best-effort breakdown similar to freq_analysis params.json.
    """
    total = int(count_parameters(model))
    active = int(count_active_parameters(model))
    return {"total_params": total, "active_params": active}


def run_acc_benchmark(
    dataset: str,
    out_root: str,
    data_root: str,
    hidden_dims: Sequence[int],
    models: Optional[Sequence[str]] = None,
    epochs: int = 50,
    soft_mask_epochs: Optional[int] = None,
    stabilize_epochs: int = 0,
    ste_epochs: int = 0,
    batch_size: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    weight_decay_dend_soma: Optional[float] = None,
    seed: int = 0,
    S_min: float = 1.0,
    S_max_list: Sequence[float] = (8.0,),
    th_len: int = 4,
    v_th: float = 1.0,
    v_reset: Optional[float] = None,
    v_pre: float = 1.0,
    cifar_mode: str = "parallel",
    T_event: int = 250,
    num_workers: int = 4,
    download: bool = True,
    device: str = "auto",
    check_every: int = 1,
    max_eval_batches: Optional[int] = None,
    lambda_ortho: float = 0.0,
    lambda_s: float = 0.0,
    exp_name: Optional[str] = None,
    timestamp: Optional[str] = None,
) -> str:
    from src.common.utils import get_device

    set_seed(seed)
    dev = get_device(device)
    if v_reset is None:
        v_reset = float(v_th)

    train_loader, test_loader, num_classes, input_dim, T = _load_dataset(
        dataset, data_root, batch_size, num_workers, download, cifar_mode=cifar_mode, T_event=T_event, seed=seed
    )

    if models is None:
        models = ALL_MODELS

    ts = str(timestamp) if timestamp is not None else now_timestamp_seoul()
    default_exp_name = f"acc_benchmark-{dataset}"
    exp_name_final = (exp_name or default_exp_name).replace(" ", "").replace("/", "-")
    out_dir = ensure_dir(os.path.join(out_root, f"{exp_name_final}_{ts}"))

    # hyperparams txt
    hp_lines = []
    hp_lines.append(f"dataset: {dataset}")
    hp_lines.append(f"models: {', '.join(models)}")
    hp_lines.append("readout: mem")
    hp_lines.append(f"hidden_dims: {list(hidden_dims)}")
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


    hp_lines.append(f"epochs: {total_e}")
    hp_lines.append(f"soft_mask_epochs: {soft_e}")
    hp_lines.append(f"stabilize_epochs: {stb_e}")
    hp_lines.append(f"ste_epochs: {ste_e}")
    hp_lines.append(f"batch_size: {batch_size}")
    hp_lines.append(f"lr: {lr}")
    hp_lines.append(f"weight_decay: {weight_decay} (AdamW; layer connection weights only)")
    hp_lines.append(
        f"weight_decay_dend_soma: {weight_decay_dend_soma} (my_R_DH_SNN W_mix only; None => follow weight_decay)"
    )
    hp_lines.append(f"seed: {seed}")
    hp_lines.append(f"S_max_list: {list(S_max_list)}")
    hp_lines.append(f"S_min: {S_min} (variable-branch lower bound)")
    hp_lines.append(f"th_len: {th_len}")
    hp_lines.append(f"v_th: {v_th} v_reset: {v_reset} (PLIF: learnable init)")
    hp_lines.append(f"v_pre: {v_pre}")
    hp_lines.append(f"cifar_mode: {cifar_mode}")
    hp_lines.append(f"T_event: {T_event}")
    hp_lines.append(f"check_every: {check_every}")
    hp_lines.append(f"max_eval_batches: {max_eval_batches}")
    hp_lines.append(f"lambda_ortho: {lambda_ortho} lambda_s: {lambda_s}")
    save_text(os.path.join(out_dir, "hyperparams.txt"), "\n".join(hp_lines) + "\n")

    # Model structure + active parameter breakdown (by type)
    param_report_lines: List[str] = []
    param_report_lines.append(f"experiment: {exp_name_final}")
    # 기록은 실제 실행에 사용된 timestamp 문자열(ts)을 사용한다.
    param_report_lines.append(f"timestamp: {ts}")
    param_report_lines.append(f"seed: {seed}")
    param_report_lines.append(f"device: {dev}")
    param_report_lines.append(f"dataset: {dataset}")
    param_report_lines.append(f"architecture: input_dim={input_dim} hidden_dims={list(hidden_dims)} num_classes={num_classes}")
    param_report_lines.append(f"S_max_list={list(S_max_list)} (also sets max branch dimension)")
    param_report_lines.append(f"S_min={S_min} (for proposed variable-branch models)")
    param_report_lines.append(f"th_len={th_len} (for D-RF variants)")
    param_report_lines.append(f"v_th={v_th} v_reset={v_reset} (PLIF: always learnable)")
    param_report_lines.append(f"v_pre={v_pre} (D-RF pre-threshold scaling)")
    param_report_lines.append("")

    results_lines = []
    results_lines.append("model\tbest_test_acc\tfinal_test_acc\tactive_params\ttotal_params")

    dendritic_models = {
        "dh-snn",
        "d-rf",
        "my-dh-snn",
        "my-r-dh-snn",
        "my-d-rf",
    }
    S_max_values = [float(x) for x in (S_max_list or (8.0,))]
    if len(S_max_values) == 0:
        S_max_values = [8.0]

    for mname in models:
        base_name = str(mname)
        mkey = base_name.lower().strip()
        smax_runs = S_max_values if mkey in dendritic_models else [S_max_values[0]]

        for smax in smax_runs:
            smax = float(smax)
            if not (smax > 0.0):
                raise ValueError(f"S_max must be > 0 (got {smax})")
            if not (1.0 <= float(S_min) <= smax):
                raise ValueError(f"Require 1 <= S_min <= S_max (got S_min={S_min}, S_max={smax})")

            br = int(derive_branch_from_S_max(smax))
            variant = f"{base_name}_Smax{float_to_tag(smax)}" if mkey in dendritic_models else base_name
            model_dir = ensure_dir(os.path.join(out_dir, variant))
            net = _build_mlp_snn(
                base_name,
                input_dim,
                hidden_dims,
                num_classes,
                branch=br,
                S_min=S_min,
                S_max=smax,
                th_len=th_len,
                v_th=v_th,
                v_reset=v_reset,
                v_pre=v_pre,
            ).to(dev)

            # Record model structure + active parameter breakdown
            try:
                param_report_lines.append(f"[{variant}] model_structure")
                dims = [input_dim] + list(hidden_dims) + [num_classes]
                param_report_lines.append("  " + " -> ".join(str(d) for d in dims))
                if mkey in dendritic_models:
                    param_report_lines.append(f"  S_max={smax} -> branch={br} (derived)")
                layer_breakdowns = []
                # FFNet exposes .layers
                for li, layer in enumerate(getattr(net, 'layers', []), start=1):
                    bd = layer_active_param_breakdown(layer)
                    layer_breakdowns.append(bd)
                    param_report_lines.append(f"  layer{li}: {layer.__class__.__name__}")
                    param_report_lines.append(format_breakdown_table(bd, prefix="    "))
                total_bd = aggregate_breakdowns(layer_breakdowns)
                param_report_lines.append("  INITIAL_ACTIVE_PARAMS_BY_TYPE")
                param_report_lines.append(format_breakdown_table(total_bd, prefix="    "))
                param_report_lines.append(f"  initial_total_params: {count_parameters(net)}")
                param_report_lines.append(f"  initial_total_active_params: {count_active_parameters(net)}")
                param_report_lines.append("")
            except Exception as e:
                param_report_lines.append(f"[{variant}] param breakdown failed: {e}")
                param_report_lines.append("")

            from src.common.optim import build_adamw

            opt, opt_info = build_adamw(
                net,
                lr=lr,
                weight_decay=float(weight_decay),
                weight_decay_dend_soma=weight_decay_dend_soma,
            )
            # Record optimizer grouping (best-effort; does not affect training)
            try:
                param_report_lines.append(f"[{variant}] optimizer")
                param_report_lines.append(f"  type: AdamW")
                param_report_lines.append(f"  lr: {lr}")
                param_report_lines.append(f"  weight_decay(layer): {opt_info.weight_decay}")
                param_report_lines.append(f"  weight_decay(dend_soma): {opt_info.weight_decay_dend_soma}")
                param_report_lines.append(f"  num_decay_layer_params: {opt_info.num_decay_layer_params}")
                param_report_lines.append(f"  num_decay_dend_soma_params: {opt_info.num_decay_dend_soma_params}")
                param_report_lines.append(f"  num_no_decay_params: {opt_info.num_no_decay_params}")
                param_report_lines.append("")
            except Exception:
                pass
            criterion = nn.CrossEntropyLoss()

            eval_epochs = []
            train_accs = []
            test_accs = []
            best_acc = -1.0

            pbar = tqdm(range(1, total_e + 1), desc=variant, total=int(total_e), leave=True)
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
                    from src.common.model_utils import harden_variable_branches_

                    harden_variable_branches_(net)
                net.train()
                for x, y in train_loader:
                    x = x.to(dev)
                    y = y.to(dev).long()

                    opt.zero_grad(set_to_none=True)
                    logits = net(x)
                    loss = criterion(logits, y)

                    if hasattr(net, "regularization_loss"):
                        loss = loss + net.regularization_loss(lambda_ortho=lambda_ortho, lambda_s=lambda_s)

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                    opt.step()

                if check_every > 0 and (epoch % check_every == 0):
                    tr_acc, _ = evaluate_classifier(net, train_loader, dev, max_batches=max_eval_batches)
                    te_acc, _ = evaluate_classifier(net, test_loader, dev, max_batches=max_eval_batches)
                    eval_epochs.append(epoch)
                    train_accs.append(tr_acc)
                    test_accs.append(te_acc)
                    best_acc = max(best_acc, te_acc)
                    pbar.set_postfix(train_acc=f"{tr_acc:.4f}", test_acc=f"{te_acc:.4f}")

            final_test = test_accs[-1] if test_accs else 0.0

            # Save acc plot
            save_line_plot(
                os.path.join(model_dir, "acc_curve.png"),
                {"train": train_accs, "test": test_accs},
                x=eval_epochs,
                xlabel="epoch",
                ylabel="acc",
                title=f"{variant} acc",
            )

            # Save distributions
            _save_model_distributions(model_dir, net, variant)

            # Params breakdown
            total_params = int(count_parameters(net))
            active_params = int(count_active_parameters(net))

            # Final (post-training) active parameter breakdown by type.
            # For variable-branch proposed models, this reflects the learned/pruned structure
            # (especially when stabilize_epochs>0 triggers hardening + s freeze).
            try:
                param_report_lines.append(f"[{variant}] final_active_param_breakdown")
                layer_breakdowns_f = []
                for li, layer in enumerate(getattr(net, 'layers', []), start=1):
                    bd_f = layer_active_param_breakdown(layer)
                    layer_breakdowns_f.append(bd_f)
                    param_report_lines.append(f"  layer{li}: {layer.__class__.__name__}")
                    param_report_lines.append(format_breakdown_table(bd_f, prefix="    "))
                total_bd_f = aggregate_breakdowns(layer_breakdowns_f)
                param_report_lines.append("  TOTAL_FINAL_ACTIVE_PARAMS_BY_TYPE")
                param_report_lines.append(format_breakdown_table(total_bd_f, prefix="    "))
                param_report_lines.append(f"  total_params: {total_params}")
                param_report_lines.append(f"  total_active_params: {active_params}")
                param_report_lines.append("")
            except Exception as e:
                param_report_lines.append(f"[{variant}] final param breakdown failed: {e}")
                param_report_lines.append("")

            results_lines.append(f"{variant}\t{best_acc:.6f}\t{final_test:.6f}\t{active_params}\t{total_params}")

            # Save model checkpoint
            torch.save({"model_state_dict": net.state_dict()}, os.path.join(model_dir, "final.pt"))

    save_text(os.path.join(out_dir, "results.txt"), "\n".join(results_lines) + "\n")

    save_text(os.path.join(out_dir, "model_structure_and_params.txt"), "\n".join(param_report_lines) + "\n")
    return out_dir
