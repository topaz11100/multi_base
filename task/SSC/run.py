import argparse
from datetime import datetime
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT))
sys.path.append(str(Path(__file__).resolve().parent))

import torch
from torch import nn, optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from task.SSC.data import DEFAULT_SSC_DATA_ROOT, make_dataloaders
from task.SSC.model import build_model
from task.SSC.util import (
    accuracy_top1,
    count_trainable_params,
    make_result_dir,
    plot_accuracies_bar,
    plot_accuracies_overlay,
    save_accuracy_txt,
    save_meta,
    seed_everything,
)


NEURON_CHOICES = ["cp", "tc", "ts", "dh-sfnn", "dh-srnn"]


def parse_args():
    parser = argparse.ArgumentParser(description="SSC neuron comparison")
    parser.add_argument("--exp-name", default="SSC", dest="exp_name")
    parser.add_argument("--neurons", default="all", help="comma list or all")
    parser.add_argument("--protocol", default="tc", choices=["tc", "dh"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")

    parser.add_argument("--data-root", default=str(DEFAULT_SSC_DATA_ROOT)) # required=True 제거
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true")

    parser.add_argument("--T", type=int, default=None)
    parser.add_argument("--max-time", type=float, default=None)
    parser.add_argument("--dt", type=float, default=None)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--backbone", choices=["ff", "rnn"], default="ff")
    parser.add_argument("--nb-units", type=int, default=700)
    parser.add_argument("--num-classes", type=int, default=35)

    parser.add_argument("--vth", type=float, default=1.0)
    parser.add_argument("--reset-mode", choices=["soft", "hard"], default="soft")
    parser.add_argument("--sg-beta", type=float, default=10.0)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--beta1", type=float, default=-0.5)
    parser.add_argument("--beta2", type=float, default=0.5)
    parser.add_argument("--tau-init", type=float, default=8.0)
    parser.add_argument("--soft-reset", type=float, default=1.0)
    parser.add_argument("--tau-s-low", type=float, default=4.0)
    parser.add_argument("--tau-s-high", type=float, default=12.0)
    parser.add_argument("--tau-d", type=float, default=8.0)

    parser.add_argument("--dh-low-n", type=float, default=2.0)
    parser.add_argument("--dh-high-n", type=float, default=6.0)
    parser.add_argument("--dh-branch-sfnn", type=int, default=4)
    parser.add_argument("--dh-branch-srnn", type=int, default=8)
    parser.add_argument("--dh-sfnn-layers", type=int, default=2)

    parser.add_argument("--readout-mode", choices=["sum", "mean", "sum_softmax"], default="sum")
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--match-params", action="store_true")
    parser.add_argument("--param-tolerance", type=float, default=0.05)
    parser.add_argument("--use-amp", action="store_true")
    return parser.parse_args()


def select_neurons(arg_val: str):
    if arg_val == "all":
        return NEURON_CHOICES
    chosen = [n.strip() for n in arg_val.split(",") if n.strip()]
    for name in chosen:
        if name not in NEURON_CHOICES:
            raise ValueError(f"Unknown neuron name {name}")
    return chosen


def adjust_hidden_for_params(neurons, args):
    reference = neurons[0]
    base_model = build_model(reference, args)
    target_params = count_trainable_params(base_model)
    results = {reference: (base_model, target_params)}
    for neuron in neurons[1:]:
        model = build_model(neuron, args)
        params = count_trainable_params(model)
        hidden = args.hidden
        if args.match_params:
            for delta in range(-16, 17, 4):
                candidate_hidden = max(8, hidden + delta)
                setattr(args, "hidden", candidate_hidden)
                candidate_model = build_model(neuron, args)
                candidate_params = count_trainable_params(candidate_model)
                if abs(candidate_params - target_params) / target_params <= args.param_tolerance:
                    model = candidate_model
                    params = candidate_params
                    break
                del candidate_model
            setattr(args, "hidden", hidden)
        results[neuron] = (model, params)
    return results


def run_epoch(model: nn.Module, loader: DataLoader, optimizer, device, use_amp: bool, train: bool = True):
    model.train(train)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler(enabled=use_amp)
    running_loss = 0.0
    running_acc = 0.0
    batches = 0
    for x, y in tqdm(loader, leave=False):
        x = x.to(device)
        y = y.to(device)
        if hasattr(model, "reset_state"):
            reset_kwargs = {}
            if "batch_size" in model.reset_state.__code__.co_varnames:
                reset_kwargs["batch_size"] = x.size(0)
            model.reset_state(**reset_kwargs)
        if train:
            optimizer.zero_grad()
            with autocast(enabled=use_amp):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if hasattr(model, "apply_masks"):
                model.apply_masks()
            elif hasattr(model, "apply_mask"):
                model.apply_mask()
        else:
            with torch.no_grad():
                logits = model(x)
                loss = criterion(logits, y)
        running_loss += loss.item()
        running_acc += accuracy_top1(logits, y)
        batches += 1
    return running_loss / max(1, batches), running_acc / max(1, batches)


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)
    result_dir = make_result_dir(args.exp_name)
    start_time = datetime.now()
    train_loader, val_loader, test_loader = make_dataloaders(args)

    neurons = select_neurons(args.neurons)
    models_info = adjust_hidden_for_params(neurons, args)

    meta_extra = {
        "start_time": start_time.isoformat(),
        "command": " ".join(
            [
                "python",
                str(Path(__file__).resolve()),
            ]
            + [f"{k}={v}" for k, v in vars(args).items()]
        ),
    }
    save_meta(result_dir, args, meta_extra)

    best_results = {}

    for neuron in neurons:
        model, params = models_info[neuron]
        model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
        best_val = -1.0
        best_epoch = -1
        for epoch in range(1, args.epochs + 1):
            epoch_desc = f"{neuron} Epoch {epoch}/{args.epochs}"
            with tqdm(total=len(train_loader), desc=epoch_desc, leave=True) as pbar:
                train_loss, train_acc = run_epoch(model, train_loader, optimizer, device, args.use_amp, train=True)
                pbar.update(len(train_loader))
            val_loss, val_acc = run_epoch(model, val_loader, optimizer, device, args.use_amp, train=False)
            scheduler.step()
            if val_acc > best_val:
                best_val = val_acc
                best_epoch = epoch
            tqdm.write(f"[{neuron}] epoch {epoch}: train_acc={train_acc:.4f} val_acc={val_acc:.4f}")
        test_loss, test_acc = run_epoch(model, test_loader, optimizer, device, args.use_amp, train=False)
        best_results[neuron] = {
            "test_acc": test_acc,
            "best_val_acc": best_val,
            "best_epoch": best_epoch,
            "trainable_params": params,
        }
        save_accuracy_txt(result_dir, neuron, best_results[neuron])

    acc_dict = {k: v["test_acc"] for k, v in best_results.items()}
    plot_accuracies_bar(acc_dict, result_dir / "acc_per_neuron.png")
    plot_accuracies_overlay(acc_dict, result_dir / "acc_overlay.png")

    end_time = datetime.now()
    meta_extra.update({"end_time": end_time.isoformat(), "elapsed": str(end_time - start_time)})
    save_meta(result_dir, args, meta_extra)


if __name__ == "__main__":
    main()
