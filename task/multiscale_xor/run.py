"""Run multiscale XOR experiments comparing four neuron models.

Example (defaults match the paper configuration):
    python run.py --device cuda
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT))
sys.path.append(str(Path(__file__).resolve().parent))

from data import generate_batch, make_eval_mask
from model import build_model
from util import (
    count_parameters,
    make_result_dir,
    plot_overlay,
    plot_single,
    save_json,
    save_text,
    set_seed,
    tune_hidden_dim,
)

PAPER_DEFAULTS = {
    "high_rate": 0.6,
    "low_rate": 0.2,
    "noise_rate": 0.01,
    "coding_time": 10,
    "remain_time": 5,
    "trials": 10,
}


def parse_delay_args(args: argparse.Namespace) -> List[int]:
    if args.delay_list:
        return [int(x) for x in args.delay_list.split(",") if x.strip()]
    delays = list(range(args.delay_min, args.delay_max + 1, args.delay_step))
    if not delays:
        raise ValueError("At least one delay value is required")
    return delays


def apply_paper_defaults(args: argparse.Namespace) -> None:
    """Override CLI args with the paper's reported settings."""

    for key, value in PAPER_DEFAULTS.items():
        current = getattr(args, key, None)
        if current is None:
            continue
        if current != value:
            print(
                f"--paper_defaults overriding {key} from {current} to {value}",
                file=sys.stderr,
            )
        setattr(args, key, value)


def compute_loss_and_acc(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, float]:
    mask = mask.to(logits.device)
    if mask.dim() == 1:
        mask_b = mask.view(1, -1).expand(logits.size(0), -1)
    elif mask.dim() == 2:
        mask_b = mask
    else:
        raise ValueError(f"Mask must be 1D or 2D, got {mask.dim()} dimensions")

    if mask_b.shape[0] != logits.size(0) or mask_b.shape[1] != logits.size(1):
        raise ValueError(
            f"Mask shape {mask_b.shape} is incompatible with logits shape {logits.shape}"
        )

    active_count = mask_b.sum().item()
    if active_count == 0:
        raise ValueError("Evaluation mask selects zero elements; check delay/time configuration")

    logits_flat = logits[mask_b]
    targets_flat = targets[mask_b]
    loss = F.cross_entropy(logits_flat, targets_flat)
    preds = logits_flat.argmax(dim=1)
    acc = (preds == targets_flat).float().mean().item()
    return loss, acc


def enforce_structural_masks(model: torch.nn.Module, check: bool = False, eps: float = 0.0) -> None:
    """Reapply any structural masks registered on modules and optionally verify."""

    for module in model.modules():
        if hasattr(module, "apply_mask"):
            module.apply_mask()
            if check:
                mask = getattr(module, "mask", None)
                dense = getattr(module, "dense", None)
                weight = getattr(dense, "weight", None) if dense is not None else None
                if mask is not None and weight is not None:
                    max_abs = (weight * (1 - mask)).abs().max().item()
                    assert max_abs <= max(eps, 0.0), f"Mask violation detected: {max_abs}"


def select_hidden_dims(
    neuron_names: Iterable[str],
    input_dim: int,
    output_dim: int,
    base_hidden: int,
    tolerance: float,
    device: torch.device,
    neuron_kwargs: Dict[str, Dict],
) -> Dict[str, Tuple[int, int]]:
    # Hidden-dim tuning is forced onto CPU to avoid GPU memory spikes.
    cpu_device = torch.device("cpu")
    builders = {
        "dh": lambda inp, hid, out, dev=cpu_device: build_model("dh", inp, hid, out, dev, neuron_kwargs.get("dh", {})),
        "tc": lambda inp, hid, out, dev=cpu_device: build_model("tc", inp, hid, out, dev, neuron_kwargs.get("tc", {})),
        "ts": lambda inp, hid, out, dev=cpu_device: build_model("ts", inp, hid, out, dev, neuron_kwargs.get("ts", {})),
        "cp": lambda inp, hid, out, dev=cpu_device: build_model("cp", inp, hid, out, dev, neuron_kwargs.get("cp", {})),
    }
    target_params = count_parameters(builders["dh"](input_dim, base_hidden, output_dim))
    hidden_dims: Dict[str, Tuple[int, int]] = {"dh": (base_hidden, target_params)}
    search_space = range(8, 257)
    for name in neuron_names:
        if name == "dh":
            continue
        tuned_hidden, tuned_params = tune_hidden_dim(
            lambda inp, hid, out, dev=cpu_device: builders[name](inp, hid, out, dev),
            target_params,
            input_dim,
            output_dim,
            base_hidden,
            search=search_space,
            tolerance=tolerance,
            device=cpu_device,
        )
        hidden_dims[name] = (tuned_hidden, tuned_params)
    return hidden_dims


def train_and_evaluate(
    neuron: str,
    delay: int,
    hidden_dim: int,
    args: argparse.Namespace,
    device: torch.device,
    neuron_kwargs: Dict,
    eval_mask: torch.Tensor,
) -> float:
    model = build_model(neuron, args.input_dim, hidden_dim, 2, device, neuron_kwargs)
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma)

    mask = eval_mask.to(device)
    mask_check = args.dh_mask_check or args.debug_mask
    enforce_structural_masks(model)
    for epoch in tqdm(range(args.epochs), desc=f"{neuron.upper()} delay={delay} train"):
        for _ in range(args.log_interval):
            x, y = generate_batch(
                batch_size=args.batch_size,
                time_steps=args.time_steps,
                channel_size=args.channel_size,
                coding_time=args.coding_time,
                remain_time=args.remain_time,
                delay=delay,
                low_rate=args.low_rate,
                high_rate=args.high_rate,
                noise_rate=args.noise_rate,
                device=device,
            )
            mask_bt = mask.unsqueeze(0).expand(x.size(0), -1)
            enforce_structural_masks(model)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss, _ = compute_loss_and_acc(logits, y, mask_bt)
            loss.backward()
            if args.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()
            enforce_structural_masks(model, check=mask_check, eps=1e-8)
        scheduler.step()

    acc_sum = 0.0
    enforce_structural_masks(model, check=mask_check, eps=1e-8)
    with torch.no_grad():
        for _ in tqdm(range(args.eval_steps), desc=f"{neuron.upper()} delay={delay} eval"):
            x, y = generate_batch(
                batch_size=args.batch_size,
                time_steps=args.time_steps,
                channel_size=args.channel_size,
                coding_time=args.coding_time,
                remain_time=args.remain_time,
                delay=delay,
                low_rate=args.low_rate,
                high_rate=args.high_rate,
                noise_rate=args.noise_rate,
                device=device,
            )
            mask_bt = mask.unsqueeze(0).expand(x.size(0), -1)
            enforce_structural_masks(model)
            logits = model(x)
            _, acc = compute_loss_and_acc(logits, y, mask_bt)
            acc_sum += acc
    return acc_sum / max(1, args.eval_steps)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multiscale XOR delay sweep")
    # Experiment
    parser.add_argument("--exp_name", type=str, default="multiscale_xor")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trials", type=int, default=10)

    # Data
    parser.add_argument("--time_steps", type=int, default=100)
    parser.add_argument("--channel_size", type=int, default=20)
    parser.add_argument("--coding_time", type=int, default=10)
    parser.add_argument("--remain_time", type=int, default=5)
    parser.add_argument("--low_rate", type=float, default=0.2)
    parser.add_argument("--high_rate", type=float, default=0.6)
    parser.add_argument("--noise_rate", type=float, default=0.01)
    parser.add_argument("--delay_list", type=str, default=None)
    parser.add_argument("--delay_min", type=int, default=10)
    parser.add_argument("--delay_max", type=int, default=10)
    parser.add_argument("--delay_step", type=int, default=10)

    # Training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=500)
    parser.add_argument("--hidden", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--lr_step", type=int, default=50)
    parser.add_argument("--lr_gamma", type=float, default=0.1)
    parser.add_argument("--clip_grad", type=float, default=20.0)
    parser.add_argument("--log_interval", type=int, default=20)
    parser.add_argument("--eval_steps", type=int, default=10)

    # Neuron selection and hyperparameters
    parser.add_argument("--neurons", nargs="+", default=["dh", "tc", "ts", "cp"], choices=["dh", "tc", "ts", "cp"])
    parser.add_argument("--param_tolerance", type=float, default=0.05)
    parser.add_argument("--dh_branch", type=int, default=1)
    parser.add_argument("--dh_tau_initializer", type=str, default="uniform")
    parser.add_argument("--dh_low_m", type=float, default=0.0)
    parser.add_argument("--dh_high_m", type=float, default=4.0)
    parser.add_argument("--dh_vth", type=float, default=1.0)
    parser.add_argument("--dh_dt", type=float, default=1.0)
    parser.add_argument("--dh_mask_check", action="store_true", help="Validate DH masks after each step (deprecated; use --debug_mask)")
    parser.add_argument("--debug_mask", action="store_true", help="Validate DH masks after each optimizer step")
    parser.add_argument("--tc_gamma", type=float, default=0.5)
    parser.add_argument("--cp_tau_init", type=float, default=0.5)
    parser.add_argument("--ts_gamma", type=float, default=0.5)
    parser.add_argument("--paper_defaults", action="store_true", help="Use hyperparameters from the paper (rates, noise, timing, trials)")

    args = parser.parse_args()
    if args.paper_defaults:
        apply_paper_defaults(args)
        print(
            "Applied paper defaults: "
            f"high_rate={args.high_rate}, low_rate={args.low_rate}, noise_rate={args.noise_rate}, "
            f"coding_time={args.coding_time}, remain_time={args.remain_time}, trials={args.trials}",
            file=sys.stderr,
        )
    args.debug_mask = args.debug_mask or args.dh_mask_check
    args.input_dim = args.channel_size * 2
    return args


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    delays = parse_delay_args(args)
    result_dir = make_result_dir(args.exp_name)

    neuron_kwargs = {
        "dh": {
            "branch": args.dh_branch,
            "tau_initializer": args.dh_tau_initializer,
            "low_m": args.dh_low_m,
            "high_m": args.dh_high_m,
            "vth": args.dh_vth,
            "dt": args.dh_dt,
        },
        "tc": {"gamma": args.tc_gamma},
        "cp": {"tau_init": args.cp_tau_init},
        "ts": {"gamma": args.ts_gamma},
    }

    hidden_dims = select_hidden_dims(args.neurons, args.input_dim, 2, args.hidden, args.param_tolerance, device, neuron_kwargs)
    hparam_log = {
        "args": vars(args),
        "hidden_dims": {k: {"hidden": v[0], "params": v[1]} for k, v in hidden_dims.items()},
    }
    save_json(result_dir / "hparams.txt", hparam_log)

    runtime_lines = [f"start: {time.asctime()}"]

    results: Dict[str, List[Tuple[int, float, float]]] = {n: [] for n in args.neurons}
    for delay in delays:
        eval_mask = make_eval_mask(args.time_steps, args.coding_time, args.remain_time, delay)
        for neuron in args.neurons:
            trial_accs = []
            for trial in range(args.trials):
                set_seed(args.seed + trial)
                acc = train_and_evaluate(
                    neuron,
                    delay,
                    hidden_dims[neuron][0],
                    args,
                    device,
                    neuron_kwargs.get(neuron, {}),
                    eval_mask,
                )
                trial_accs.append(acc)
            mean_acc = float(torch.tensor(trial_accs).mean().item())
            std_acc = float(torch.tensor(trial_accs).std(unbiased=False).item())
            results[neuron].append((delay, mean_acc, std_acc))
            save_text(
                result_dir / f"acc_{neuron}.txt",
                "\n".join(
                    ["delay\tmean_acc\tstd_acc"]
                    + [f"{d}\t{m:.4f}\t{s:.4f}" for d, m, s in results[neuron]]
                ),
            )
            plot_single(
                neuron,
                [d for d, _, _ in results[neuron]],
                [m for _, m, _ in results[neuron]],
                result_dir / f"acc_{neuron}.png",
            )

    overlay_data = {n: [m for _, m, _ in vals] for n, vals in results.items()}
    plot_overlay(overlay_data, delays, result_dir / "acc_overlay.png")

    runtime_lines.append(f"end: {time.asctime()}")
    save_text(result_dir / "runtime.txt", "\n".join(runtime_lines))


if __name__ == "__main__":
    main()
