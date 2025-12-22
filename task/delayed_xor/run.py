import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT))
sys.path.append(str(Path(__file__).resolve().parent))

from data import generate_batch
from model import get_model
from util import (
    SaveManager,
    count_parameters,
    dry_run,
    format_settings,
    set_seed,
    tune_hidden_dim,
)


def compute_loss_and_acc(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, float]:
    if mask.sum() == 0:
        return torch.tensor(0.0, device=logits.device, requires_grad=True), 0.0
    logits_flat = logits[mask]
    targets_flat = targets[mask]
    loss = F.cross_entropy(logits_flat, targets_flat)
    preds = logits_flat.argmax(dim=1)
    acc = (preds == targets_flat).float().mean().item()
    return loss, acc


def train_one_delay(
    model_name: str,
    hidden_dim: int,
    delay: int,
    args,
    device: torch.device,
) -> Tuple[float, int]:
    model = get_model(model_name, args.channel_size, hidden_dim, 2, device)
    optimizer = Adam(model.parameters(), lr=args.lr)

    time_steps = args.time_steps or (delay + args.coding_time * 2 + args.tail_time)

    if args.dry_run and args.verbose:
        batch = generate_batch(
            batch_size=min(2, args.batch_size),
            delay=delay,
            time_steps=time_steps,
            channel_size=args.channel_size,
            coding_time=args.coding_time,
            noise_rate=args.noise_rate,
        )[0]
        dry_run(model, batch.to(device))

    for epoch in tqdm(range(args.epochs), desc=f"{model_name.upper()} Delay={delay}"):
        for _ in range(args.steps_per_epoch):
            data, target, mask = generate_batch(
                batch_size=args.batch_size,
                delay=delay,
                time_steps=time_steps,
                channel_size=args.channel_size,
                coding_time=args.coding_time,
                noise_rate=args.noise_rate,
            )
            data, target, mask = data.to(device), target.to(device), mask.to(device)
            optimizer.zero_grad()
            logits = model(data)
            loss, _ = compute_loss_and_acc(logits, target, mask)
            loss.backward()
            optimizer.step()

    acc_sum = 0.0
    eval_batches = 0
    with torch.no_grad():
        for _ in range(args.eval_steps):
            data, target, mask = generate_batch(
                batch_size=args.batch_size,
                delay=delay,
                time_steps=time_steps,
                channel_size=args.channel_size,
                coding_time=args.coding_time,
                noise_rate=args.noise_rate,
            )
            data, target, mask = data.to(device), target.to(device), mask.to(device)
            logits = model(data)
            _, acc = compute_loss_and_acc(logits, target, mask)
            acc_sum += acc
            eval_batches += 1
    final_acc = acc_sum / max(1, eval_batches)
    return final_acc, count_parameters(model)


def prepare_hidden_dims(args, device: torch.device) -> Dict[str, int]:
    base_hidden = args.hidden_dim
    cpu = torch.device("cpu")
    target_params = count_parameters(get_model("dh", args.channel_size, base_hidden, 2, cpu))

    builders = {
        "dh": lambda in_d, h, out_d, dev: get_model("dh", in_d, h, out_d, dev),
        "cp": lambda in_d, h, out_d, dev: get_model("cp", in_d, h, out_d, dev),
        "tc": lambda in_d, h, out_d, dev: get_model("tc", in_d, h, out_d, dev),
        "ts": lambda in_d, h, out_d, dev: get_model("ts", in_d, h, out_d, dev),
    }

    hidden_dims = {"dh": base_hidden}
    for name in ["cp", "tc", "ts"]:
        tuned = tune_hidden_dim(
            builders[name],
            target_params,
            args.channel_size,
            2,
            base_hidden,
            device=cpu,
        )
        hidden_dims[name] = tuned
    if args.verbose:
        print("Parameter targets:", target_params)
        for k, h in hidden_dims.items():
            params = count_parameters(builders[k](args.channel_size, h, 2, cpu))
            print(f"{k.upper()} hidden={h} params={params}")
    return hidden_dims


def main():
    parser = argparse.ArgumentParser(description="Delayed XOR comparison")
    parser.add_argument("--model", choices=["all", "dh", "cp", "tc", "ts"], default="all")
    parser.add_argument("--delay_T", type=int, default=400)
    parser.add_argument("--delay_T_delta", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--steps_per_epoch", type=int, default=20)
    parser.add_argument("--eval_steps", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--hidden_dim", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--channel_size", type=int, default=20)
    parser.add_argument("--coding_time", type=int, default=10)
    parser.add_argument("--noise_rate", type=float, default=0.01)
    parser.add_argument("--tail_time", type=int, default=2)
    parser.add_argument("--time_steps", type=int, default=None)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    save_mgr = SaveManager(root=str(ROOT / "result"))
    save_mgr.log_settings(format_settings(args))
    print(f"Saving outputs to: {save_mgr.describe()}")

    delays = list(range(0, args.delay_T + 1, args.delay_T_delta))
    model_names = [args.model] if args.model != "all" else ["dh", "cp", "tc", "ts"]

    hidden_dims = prepare_hidden_dims(args, device)

    results: Dict[str, List[Tuple[int, float]]] = {name: [] for name in model_names}

    for model_name in model_names:
        for delay in delays:
            acc, params = train_one_delay(model_name, hidden_dims[model_name], delay, args, device)
            results[model_name].append((delay, acc))
            save_mgr.append_log(model_name, f"delay={delay}, acc={acc:.4f}, params={params}")
        save_mgr.save_plot(model_name, results[model_name])

    save_mgr.save_combined(results)
    print("Finished. Result root:", save_mgr.describe())


if __name__ == "__main__":
    main()
