import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from task.SHD.data import get_shd_loaders  # noqa: E402
from task.SHD.model import build_model  # noqa: E402
from task.SHD.util import count_effective_params, dump_text, make_results_dir, plot_accuracy_curves, set_seed  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Unified SHD experiment runner")
    parser.add_argument("--exp_name", type=str, default="shd_experiment", help="experiment name")
    parser.add_argument("--models", type=str, nargs="*", default=["cp", "tc", "ts", "dh_sfnn", "dh_srnn"], help="models to run")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "adamw", "sgd"])
    parser.add_argument("--wd", type=float, default=0.0)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--lr_decay_every", type=int, default=10)
    parser.add_argument("--lr_decay_factor", type=float, default=0.8)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--out_dim", type=int, default=20)
    parser.add_argument("--in_dim", type=int, default=700)
    parser.add_argument("--T", type=int, default=250)
    parser.add_argument("--max_time", type=float, default=1.4)
    parser.add_argument("--delay", type=int, default=0)
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--hard_reset", action="store_true")
    parser.add_argument("--surrogate_beta", type=float, default=10.0)
    parser.add_argument("--threshold", type=float, default=1.0)
    parser.add_argument("--tc_tau_s_low", type=float, default=4.0)
    parser.add_argument("--tc_tau_s_high", type=float, default=12.0)
    parser.add_argument("--tc_tau_d", type=float, default=8.0)
    parser.add_argument("--tc_beta1", type=float, default=-0.5)
    parser.add_argument("--tc_beta2", type=float, default=0.5)
    parser.add_argument("--tc_gamma", type=float, default=0.5)
    parser.add_argument("--dh_dt", type=float, default=1.0)
    parser.add_argument("--dh_vth", type=float, default=1.0)
    parser.add_argument("--dh_low_n", type=float, default=2.0)
    parser.add_argument("--dh_high_n", type=float, default=6.0)
    parser.add_argument("--dh_branch_sfnn", type=int, default=4)
    parser.add_argument("--dh_branch_srnn", type=int, default=8)
    parser.add_argument("--dh_scheduler_step", type=int, default=20)
    parser.add_argument("--dh_scheduler_gamma", type=float, default=0.5)
    parser.add_argument("--dh_tau_lr_mult", type=float, default=2.0)
    parser.add_argument("--dh_sfnn_layers", type=int, default=2, choices=[1, 2])
    parser.add_argument("--data_root", type=str, default="../../ssd_ssh_data/")
    parser.add_argument("--param_tolerance_ratio", type=float, default=0.05)
    parser.add_argument("--enforce_param_match", type=int, default=1)
    parser.add_argument("--readout_mode", type=str, default="sum_logits", choices=["sum_logits", "sum_softmax"])
    parser.add_argument("--warmup_steps", type=int, default=10)
    parser.add_argument("--cache_dense", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    return parser.parse_args()


def build_optimizer(model, args, model_name: str):
    params = []
    if model_name.startswith("dh"):
        base_params = []
        tau_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if "tau" in name:
                tau_params.append(param)
            else:
                base_params.append(param)
        params = [
            {"params": base_params, "lr": args.lr},
            {"params": tau_params, "lr": args.lr * args.dh_tau_lr_mult},
        ]
    else:
        params = [{"params": model.parameters(), "lr": args.lr}]

    if args.optimizer == "adam":
        return torch.optim.Adam(params, lr=args.lr, weight_decay=args.wd)
    if args.optimizer == "adamw":
        return torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
    return torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.wd)


def build_scheduler(optimizer, args, model_name: str):
    if model_name.startswith("dh"):
        return StepLR(optimizer, step_size=args.dh_scheduler_step, gamma=args.dh_scheduler_gamma)
    return StepLR(optimizer, step_size=args.lr_decay_every, gamma=args.lr_decay_factor)


def accuracy(preds: torch.Tensor, target: torch.Tensor) -> float:
    pred_labels = preds.argmax(dim=1)
    correct = (pred_labels == target).sum().item()
    return correct / len(target)


def train_one_epoch(model, loader, criterion, optimizer, device, model_name: str, args):
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    batches = 0
    pbar = tqdm(loader, desc=f"train-{model_name}")
    for xb, yb in pbar:
        xb = xb.to(device)
        yb = yb.to(device)
        if hasattr(model, "reset_state"):
            model.reset_state(len(yb))
        optimizer.zero_grad()
        outputs = model(xb)
        loss = criterion(outputs, yb)
        loss.backward()
        optimizer.step()
        if hasattr(model, "apply_masks"):
            model.apply_masks()
        with torch.no_grad():
            batch_acc = accuracy(outputs, yb)
        total_loss += loss.item()
        total_acc += batch_acc
        batches += 1
        pbar.set_postfix({"loss": loss.item(), "acc": batch_acc})
    return total_acc / max(batches, 1), total_loss / max(batches, 1)


def evaluate(model, loader, criterion, device, model_name: str):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    batches = 0
    pbar = tqdm(loader, desc=f"eval-{model_name}")
    with torch.no_grad():
        for xb, yb in pbar:
            xb = xb.to(device)
            yb = yb.to(device)
            if hasattr(model, "reset_state"):
                model.reset_state(len(yb))
            outputs = model(xb)
            loss = criterion(outputs, yb)
            batch_acc = accuracy(outputs, yb)
            total_loss += loss.item()
            total_acc += batch_acc
            batches += 1
            pbar.set_postfix({"loss": loss.item(), "acc": batch_acc})
            if hasattr(model, "apply_masks"):
                model.apply_masks()
    return total_acc / max(batches, 1), total_loss / max(batches, 1)


def main():
    args = parse_args()
    device = torch.device(args.device)
    set_seed(args.seed)

    train_loader, test_loader = get_shd_loaders(
        batch_size=args.batch_size,
        T=args.T,
        max_time=args.max_time,
        in_dim=args.in_dim,
        seed=args.seed,
        num_workers=args.num_workers,
        data_root=args.data_root,
        cache_dense=bool(args.cache_dense),
        device=device,
    )

    models = {}
    param_counts = {}
    for name in args.models:
        model = build_model(name, args).to(device)
        models[name] = model
        param_counts[name] = count_effective_params(model)
        print(f"Model {name}: effective params = {param_counts[name]}")

    if args.enforce_param_match:
        max_p = max(param_counts.values())
        min_p = min(param_counts.values())
        tol = max_p * args.param_tolerance_ratio
        if max_p - min_p > tol:
            raise RuntimeError("Effective parameter counts differ beyond tolerance")

    histories = {name: {"train_acc": [], "test_acc": [], "best_test": 0.0, "best_epoch": -1} for name in models}
    start_time = datetime.now()
    for name, model in models.items():
        optimizer = build_optimizer(model, args, name)
        scheduler = build_scheduler(optimizer, args, name)
        criterion = nn.CrossEntropyLoss()
        for epoch in range(1, args.epochs + 1):
            train_acc, train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, name, args)
            train_loader.reset()
            test_acc, test_loss = evaluate(model, test_loader, criterion, device, name)
            test_loader.reset()
            scheduler.step()
            histories[name]["train_acc"].append(train_acc)
            histories[name]["test_acc"].append(test_acc)
            if test_acc > histories[name]["best_test"]:
                histories[name]["best_test"] = test_acc
                histories[name]["best_epoch"] = epoch
            print(
                f"{name} epoch {epoch}/{args.epochs} | train_acc {train_acc:.4f} loss {train_loss:.4f} | test_acc {test_acc:.4f}"
            )
    end_time = datetime.now()

    results_dir = make_results_dir("result", args.exp_name, timestamp=end_time)
    meta = {
        "exp_name": args.exp_name,
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "duration_sec": (end_time - start_time).total_seconds(),
        "device": args.device,
        "seed": args.seed,
        "delay": args.delay,
    }
    dump_text(results_dir, "meta.txt", json.dumps(meta, indent=2))
    dump_text(results_dir, "hparams.txt", json.dumps(vars(args), indent=2))

    for name, history in histories.items():
        lines = [
            f"best_test_acc={history['best_test']:.4f}",
            f"best_epoch={history['best_epoch']}",
            f"effective_params={param_counts[name]}",
        ]
        dump_text(results_dir, f"accuracy_{name}.txt", lines)

    plot_accuracy_curves(histories, results_dir)


if __name__ == "__main__":
    main()
