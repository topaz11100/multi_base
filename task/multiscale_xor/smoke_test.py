"""Lightweight smoke tests for all multiscale XOR neuron variants.

Runs a single forward/backward pass per neuron on a small batch. Passes if
no exceptions are raised.
"""
from __future__ import annotations

import argparse

import torch

from data import generate_batch, make_eval_mask
from model import build_model
from run import compute_loss_and_acc, enforce_structural_masks
from util import set_seed


def run_smoke(device: torch.device) -> None:
    neuron_kwargs = {
        "dh": {"branch": 1, "tau_initializer": "uniform", "low_m": 0.0, "high_m": 4.0, "vth": 1.0, "dt": 1.0},
        "tc": {"gamma": 0.5},
        "cp": {"tau_init": 0.5},
        "ts": {"gamma": 0.5},
    }
    mask = make_eval_mask(time_steps=100, coding_time=10, remain_time=5, delay=10).to(device)

    for neuron in ["dh", "tc", "ts", "cp"]:
        print(f"Running {neuron} smoke test on {device}")
        model = build_model(neuron, input_dim=40, hidden_dim=16, output_dim=2, device=device, neuron_kwargs=neuron_kwargs.get(neuron, {}))
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

        x, y = generate_batch(
            batch_size=8,
            time_steps=100,
            channel_size=20,
            coding_time=10,
            remain_time=5,
            delay=10,
            low_rate=0.2,
            high_rate=0.6,
            noise_rate=0.01,
            device=device,
        )

        enforce_structural_masks(model)
        logits = model(x)
        loss, _ = compute_loss_and_acc(logits, y, mask.unsqueeze(0).expand(x.size(0), -1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        enforce_structural_masks(model, check=True, eps=1e-8)
        print(f"{neuron} PASS")


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test for multiscale XOR models")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    run_smoke(device)
if __name__ == "__main__":
    main()
