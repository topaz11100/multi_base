from __future__ import annotations

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT))
sys.path.append(str(Path(__file__).resolve().parent))

import argparse
import json
from datetime import datetime
from typing import Dict, List

import torch
from torch import nn
from torch.optim import Adam, AdamW, SGD
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm

from data import make_dataloaders
from model import ModelConfig, build_model
from util import (
    accuracy_top1,
    count_params,
    make_result_dir,
    plot_curves,
    plot_final_bar,
    reset_states,
    save_json,
    save_meta,
    save_text,
    set_seed,
)


NEURON_LIST = ["cp", "tc", "ts", "dh-sfnn", "dh-srnn"]


def parse_args():
    parser = argparse.ArgumentParser(description="S-MNIST comparison pipeline")
    parser.add_argument("--exp-name", default="s-mnist")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--task", choices=["SMNIST", "PSMNIST"], default="SMNIST")
    parser.add_argument("--arch", choices=["ff", "fb"], default="ff")
    parser.add_argument("--in-dim", type=int, default=1, choices=[1, 4, 8])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--lr-milestones", nargs="*", type=int, default=[60, 80])
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--neurons", type=str, default="all")
    parser.add_argument("--hidden", type=str, default="64,256,256")
    parser.add_argument("--readout-mode", choices=["sum", "mean"], default="sum")
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--match-params", type=int, default=1)
    parser.add_argument("--param-tolerance", type=float, default=0.05)
    parser.add_argument("--target-neuron", type=str, default="cp")
    parser.add_argument("--data-root", type=str, default="./mnist_data")
    parser.add_argument("--val-split", type=float, default=0.0)
    parser.add_argument("--neuron-kwargs", type=str, default="{}", help="JSON string for neuron-specific kwargs")
    # TC/TS specific convenience
    parser.add_argument("--threshold", type=float, default=1.0)
    parser.add_argument("--tc-beta1", type=float, default=-0.5)
    parser.add_argument("--tc-beta2", type=float, default=0.5)
    parser.add_argument("--tc-gamma", type=float, default=0.5)
    parser.add_argument("--tc-hard-reset", action="store_true")
    # DH options
    parser.add_argument("--dh-branch-sfnn", type=int, default=4)
    parser.add_argument("--dh-branch-srnn", type=int, default=8)
    parser.add_argument("--dh-low-n", type=float, default=2.0)
    parser.add_argument("--dh-high-n", type=float, default=6.0)
    parser.add_argument("--dh-vth", type=float, default=1.0)
    parser.add_argument("--dh-dt", type=float, default=1.0)
    parser.add_argument("--dh-layers", type=int, default=2)
    return parser.parse_args()


def parse_hidden(hidden_str: str) -> List[int]:
    try:
        return [int(x) for x in hidden_str.split(",")]
    except Exception as exc:  # pragma: no cover - defensive
        raise argparse.ArgumentTypeError("hidden must be comma-separated ints") from exc


def parse_neuron_kwargs(args) -> Dict:
    base_kwargs: Dict = json.loads(args.neuron_kwargs)
    base_kwargs.setdefault("tc", {})
    base_kwargs.setdefault("ts", {})
    base_kwargs.setdefault("cp", {})
    base_kwargs.setdefault("dh-sfnn", {})
    base_kwargs.setdefault("dh-srnn", {})
    base_kwargs["tc"].setdefault("v_threshold", args.threshold)
    base_kwargs["tc"].setdefault("decay_factor", torch.tensor([[args.tc_beta1, args.tc_beta2]]))
    base_kwargs["tc"].setdefault("gamma", args.tc_gamma)
    base_kwargs["tc"].setdefault("hard_reset", args.tc_hard_reset)
    base_kwargs["ts"].setdefault("v_threshold", args.threshold)
    base_kwargs["ts"].setdefault("decay_factor", torch.tensor([args.tc_beta1, args.tc_beta2, args.tc_gamma, args.tc_gamma]))
    base_kwargs["ts"].setdefault("gamma", args.tc_gamma)
    base_kwargs["cp"].setdefault("v_threshold", args.threshold)
    base_kwargs["dh-sfnn"].setdefault("branch", args.dh_branch_sfnn)
    base_kwargs["dh-sfnn"].setdefault("low_n", args.dh_low_n)
    base_kwargs["dh-sfnn"].setdefault("high_n", args.dh_high_n)
    base_kwargs["dh-sfnn"].setdefault("vth", args.dh_vth)
    base_kwargs["dh-sfnn"].setdefault("dt", args.dh_dt)
    base_kwargs["dh-sfnn"].setdefault("layers", args.dh_layers)
    base_kwargs["dh-srnn"].setdefault("branch", args.dh_branch_srnn)
    base_kwargs["dh-srnn"].setdefault("low_n", args.dh_low_n)
    base_kwargs["dh-srnn"].setdefault("high_n", args.dh_high_n)
    base_kwargs["dh-srnn"].setdefault("vth", args.dh_vth)
    base_kwargs["dh-srnn"].setdefault("dt", args.dh_dt)
    return base_kwargs


def scale_hidden(hidden: List[int], scale: float) -> List[int]:
    return [max(1, int(round(h * scale))) for h in hidden]


def build_with_param_match(neuron: str, base_hidden: List[int], target_params: int, args, neuron_kwargs: Dict):
    tolerance = args.param_tolerance
    device = torch.device("cpu")
    best_model = None
    best_hidden = base_hidden
    best_param = None
    scales = [1.0] + [0.5 + 0.1 * i for i in range(1, 11)]  # 0.6..1.5
    for scale in scales:
        hidden_dims = scale_hidden(base_hidden, scale)
        cfg = ModelConfig(
            in_dim=args.in_dim,
            hidden_dims=hidden_dims,
            out_dim=10,
            arch=args.arch,
            readout_mode=args.readout_mode,
            warmup=args.warmup,
        )
        model = build_model(neuron, cfg, neuron_kwargs)
        params = count_params(model)
        if best_param is None or abs(params - target_params) < abs(best_param - target_params):
            best_param = params
            best_model = model
            best_hidden = hidden_dims
        if abs(params - target_params) / target_params <= tolerance:
            break
    if best_model is None:
        raise RuntimeError("Failed to build model")
    return best_model, best_param, best_hidden


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer,
    device,
    neuron: str,
    non_blocking: bool = False,
    scaler: GradScaler | None = None,
    amp_enabled: bool = False,
):
    model.train()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_acc = 0.0
    count = 0
    for x, y in tqdm(loader, desc=f"train-{neuron}"):
        x = x.to(device, non_blocking=non_blocking)
        y = y.to(device, non_blocking=non_blocking)
        reset_states(model, batch_size=x.size(0), device=device)
        optimizer.zero_grad(set_to_none=True)
        if amp_enabled and scaler is not None:
            with autocast():
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
        if hasattr(model, "apply_masks"):
            model.apply_masks()
        batch_size = y.size(0)
        total_loss += loss.item() * batch_size
        total_acc += accuracy_top1(logits, y) * batch_size
        count += batch_size
    return total_loss / max(count, 1), total_acc / max(count, 1)


def evaluate(model: nn.Module, loader, device, neuron: str, non_blocking: bool = False, amp_enabled: bool = False):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_acc = 0.0
    count = 0
    with torch.no_grad():
        for x, y in tqdm(loader, desc=f"eval-{neuron}"):
            x = x.to(device, non_blocking=non_blocking)
            y = y.to(device, non_blocking=non_blocking)
            reset_states(model, batch_size=x.size(0), device=device)
            if amp_enabled:
                with autocast():
                    logits = model(x)
                    loss = criterion(logits, y)
            else:
                logits = model(x)
                loss = criterion(logits, y)
            batch_size = y.size(0)
            total_loss += loss.item() * batch_size
            total_acc += accuracy_top1(logits, y) * batch_size
            count += batch_size
    return total_loss / max(count, 1), total_acc / max(count, 1)


def main():
    args = parse_args()
    args.hidden = parse_hidden(args.hidden)
    set_seed(args.seed, deterministic=args.deterministic)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available. Use --device auto or cpu instead.")

    # Auto pinning for GPU unless explicitly disabled
    args.pin_memory = args.pin_memory or device.type == "cuda"
    non_blocking = args.pin_memory and device.type == "cuda"
    args.device = device.type

    neuron_kwargs = parse_neuron_kwargs(args)

    train_loader, val_loader, test_loader = make_dataloaders(args)

    neuron_list = NEURON_LIST if args.neurons == "all" else [n.strip() for n in args.neurons.split(",")]

    target_cfg = ModelConfig(
        in_dim=args.in_dim,
        hidden_dims=args.hidden,
        out_dim=10,
        arch=args.arch,
        readout_mode=args.readout_mode,
        warmup=args.warmup,
    )
    target_model = build_model(args.target_neuron, target_cfg, neuron_kwargs)
    target_params = count_params(target_model)

    out_dir = make_result_dir(args.exp_name)
    start_time = datetime.now()

    acc_histories = {}
    final_accs = {}
    per_neuron_info = {}

    amp_enabled = args.amp and device.type == "cuda"
    scaler = GradScaler(enabled=amp_enabled)

    for neuron in neuron_list:
        model, param_count, chosen_hidden = (
            build_with_param_match(neuron, args.hidden, target_params, args, neuron_kwargs)
            if args.match_params
            else (
                build_model(
                    neuron,
                    ModelConfig(
                        in_dim=args.in_dim,
                        hidden_dims=args.hidden,
                        out_dim=10,
                        arch=args.arch,
                        readout_mode=args.readout_mode,
                        warmup=args.warmup,
                    ),
                    neuron_kwargs,
                ),
                None,
                args.hidden,
            )
        )
        model.to(device)
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = MultiStepLR(optimizer, milestones=args.lr_milestones, gamma=0.1)

        train_curve: List[float] = []
        test_curve: List[float] = []
        train_loss_curve: List[float] = []
        test_loss_curve: List[float] = []
        best_acc = 0.0
        best_epoch = 0

        for epoch in tqdm(range(1, args.epochs + 1), desc=f"epoch-{neuron}"):
            train_loss, train_acc = train_one_epoch(
                model,
                train_loader,
                optimizer,
                device,
                neuron,
                non_blocking=non_blocking,
                scaler=scaler,
                amp_enabled=amp_enabled,
            )
            test_loss, test_acc = evaluate(
                model, test_loader, device, neuron, non_blocking=non_blocking, amp_enabled=amp_enabled
            )
            scheduler.step()
            train_curve.append(train_acc)
            test_curve.append(test_acc)
            train_loss_curve.append(train_loss)
            test_loss_curve.append(test_loss)
            if test_acc > best_acc:
                best_acc = test_acc
                best_epoch = epoch

        acc_histories[neuron] = test_curve
        final_accs[neuron] = best_acc
        per_neuron_info[neuron] = {
            "param_count": param_count if param_count is not None else count_params(model),
            "hidden_dims": chosen_hidden,
            "best_test_acc": best_acc,
            "best_epoch": best_epoch,
            "final_test_acc": test_curve[-1] if test_curve else 0.0,
            "final_train_acc": train_curve[-1] if train_curve else 0.0,
            "final_train_loss": train_loss_curve[-1] if train_loss_curve else 0.0,
            "final_test_loss": test_loss_curve[-1] if test_loss_curve else 0.0,
            "train_acc_curve": train_curve,
            "test_acc_curve": test_curve,
            "train_loss_curve": train_loss_curve,
            "test_loss_curve": test_loss_curve,
            "delay": None,
        }

        save_text(
            out_dir / f"acc_{neuron}.txt",
            "\n".join(
                [
                    f"neuron: {neuron}",
                    f"param_count: {per_neuron_info[neuron]['param_count']}",
                    f"hidden_dims: {per_neuron_info[neuron]['hidden_dims']}",
                    f"best_test_acc: {best_acc}",
                    f"best_epoch: {best_epoch}",
                    f"final_train_acc: {per_neuron_info[neuron]['final_train_acc']}",
                    f"final_test_acc: {per_neuron_info[neuron]['final_test_acc']}",
                    f"final_train_loss: {per_neuron_info[neuron]['final_train_loss']}",
                    f"final_test_loss: {per_neuron_info[neuron]['final_test_loss']}",
                    "delay: null",
                ]
            ),
        )
        plot_curves({neuron: test_curve}, out_dir / f"curve_{neuron}.png")
        plot_curves({"train": train_loss_curve, "test": test_loss_curve}, out_dir / f"loss_{neuron}.png", ylabel="Loss")

    plot_curves(acc_histories, out_dir / "curves_all.png")
    plot_final_bar(final_accs, out_dir / "final_acc_bar.png")

    end_time = datetime.now()
    save_meta(out_dir, start_time, end_time, args, {f"params_{k}": v["param_count"] for k, v in per_neuron_info.items()})
    save_json(
        out_dir / "hyperparams.json",
        {
            "args": vars(args),
            "neuron_kwargs": {k: str(v) for k, v in neuron_kwargs.items()},
            "target_params": target_params,
            "per_neuron": per_neuron_info,
        },
    )


if __name__ == "__main__":
    main()
