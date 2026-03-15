from __future__ import annotations

import argparse
import os
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_ROOT = os.path.abspath(os.path.join(THIS_DIR, "../../.."))
sys.path.insert(0, PROJ_ROOT)

from src.common.long_term_mem_driver import ALL_MODELS, run_long_term_mem_xor  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("basic_long_term_mem - multiscale_XOR (single-neuron)")

    p.add_argument("--gpu", type=int, default=0, help="CUDA GPU index")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--exp_name", type=str, default=None, help="override experiment name (result folder prefix)")
    p.add_argument("--timestamp", type=str, default=None, help="override YYmmdd_HHMMSS timestamp")
    p.add_argument("--out_root", type=str, default=None)
    p.add_argument("--data_root", type=str, default=None)
    p.add_argument("--models", nargs="+", default=ALL_MODELS)
    p.add_argument("--hidden", nargs="*", type=int, default=[256])

    # training
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--soft_mask_epochs", type=int, default=None, help="stage A epochs (soft mask learning)")
    p.add_argument("--stabilize_epochs", type=int, default=0, help="stage B epochs after hardening branches")
    p.add_argument("--ste_epochs", type=int, default=0, help="STE epochs at end of stage A (forward hard / backward soft)")
    p.add_argument("--steps_per_epoch", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=500)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument(
        "--weight_decay",
        type=float,
        default=0.0,
        help="AdamW weight decay for layer connection weights only (0 disables)",
    )
    p.add_argument(
        "--weight_decay_dend_soma",
        type=float,
        default=None,
        help="my_R_DH_SNN only: weight decay for dendrite->soma weights (W_mix). None => follow --weight_decay",
    )
    p.add_argument("--check_every", type=int, default=1)
    p.add_argument("--eval_batches", type=int, default=20)

    # neuron structure (branch count is controlled ONLY by S_min/S_max)
    p.add_argument("--S_min", type=float, default=1.0, help="lower bound for continuous structural param s")
    p.add_argument("--S_max", type=float, default=8.0, help="upper bound for continuous s (also sets max branch dimension)")
    p.add_argument("--th_len", type=int, default=4)
    p.add_argument("--v_th", type=float, default=1.0)
    p.add_argument("--v_reset", type=float, default=0.0, help="reset init (<=0: use v_th)")
    p.add_argument("--v_pre", type=float, default=1.0)
    p.add_argument("--lambda_ortho", type=float, default=0.0)
    p.add_argument("--lambda_s", type=float, default=0.0)

    # multiscale XOR task params
    p.add_argument("--time_steps", type=int, default=100)
    p.add_argument("--channel_size", type=int, default=20)
    p.add_argument("--coding_time", type=int, default=10)
    p.add_argument("--remain_time", type=int, default=5)
    p.add_argument("--start_time", type=int, default=10)

    # shared signal stats
    p.add_argument("--noise_rate", type=float, default=0.01)
    p.add_argument("--rate_low", type=float, default=0.2)
    p.add_argument("--rate_high", type=float, default=0.6)

    return p


def main():
    args = build_parser().parse_args()
    out_root = args.out_root or os.path.join(PROJ_ROOT, "result")

    run_long_term_mem_xor(
        task="multiscale_XOR",
        models=args.models,
        out_root=out_root,
        data_root=args.data_root or os.path.join(PROJ_ROOT, "data"),
        exp_name=args.exp_name,
        timestamp=args.timestamp,
        seed=args.seed,
        device=f"cuda:{args.gpu}",
        epochs=args.epochs,
        soft_mask_epochs=args.soft_mask_epochs,
        stabilize_epochs=args.stabilize_epochs,
        ste_epochs=args.ste_epochs,
        steps_per_epoch=args.steps_per_epoch,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        weight_decay_dend_soma=args.weight_decay_dend_soma,
        check_every=args.check_every,
        eval_batches=args.eval_batches,
        hidden=args.hidden,
        S_min=args.S_min,
        S_max=args.S_max,
        th_len=args.th_len,
        v_th=args.v_th,
        v_reset=None if args.v_reset <= 0 else args.v_reset,
        v_pre=args.v_pre,
        lambda_ortho=args.lambda_ortho,
        lambda_s=args.lambda_s,
        multi_time_steps=args.time_steps,
        multi_channel_size=args.channel_size,
        multi_coding_time=args.coding_time,
        multi_remain_time=args.remain_time,
        multi_start_time=args.start_time,
        noise_rate=args.noise_rate,
        rate_low=args.rate_low,
        rate_high=args.rate_high,
    )


if __name__ == "__main__":
    main()
