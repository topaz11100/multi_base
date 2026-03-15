import argparse
import os
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_ROOT = os.path.abspath(os.path.join(THIS_DIR, "../../.."))
sys.path.insert(0, PROJ_ROOT)

from src.common.benchmark_driver import ALL_MODELS, run_acc_benchmark  # noqa: E402


def main():
    p = argparse.ArgumentParser(description="Accuracy benchmark: s-CIFAR10 (all neuron models)")
    p.add_argument("--out_root", type=str, default=os.path.join(PROJ_ROOT, "result"))
    p.add_argument("--data_root", type=str, default=os.path.join(PROJ_ROOT, "data"))
    # unified CLI style (user request): --models lif plif ... (nargs+)
    # keep compatibility with the special token "all".
    p.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=["all"],
        help="one or more models (e.g., --models lif plif). Use --models all to run all models.",
    )
    # Readout is fixed to membrane potential (mem).
    p.add_argument("--cifar_mode", type=str, default="parallel", choices=["parallel", "serial"])
    p.add_argument("--hidden", type=int, nargs="+", default=[256], help="hidden layer sizes, e.g. --hidden 256 256")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--soft_mask_epochs", type=int, default=None, help="stage A epochs (soft mask learning)")
    p.add_argument("--stabilize_epochs", type=int, default=0, help="stage B epochs after hardening branches")
    p.add_argument("--ste_epochs", type=int, default=0, help="STE epochs at end of stage A (forward hard / backward soft)")
    p.add_argument("--batch_size", type=int, default=128)
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
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--exp_name", type=str, default=None, help="override experiment name (result folder prefix)")
    p.add_argument("--timestamp", type=str, default=None, help="override YYmmdd_HHMMSS timestamp")

    # Branch count is controlled ONLY by S_min/S_max (no separate dendritic arg)
    p.add_argument("--S_min", type=float, default=1.0, help="lower bound for continuous structural param s")
    p.add_argument(
        "--S_max",
        type=float,
        nargs="+",
        default=[8.0],
        help="one or more S_max values (also sets max branch dimension). Dendritic models run all values.",
    )

    p.add_argument("--th_len", type=int, default=4)
    p.add_argument("--v_th", type=float, default=1.0)
    p.add_argument("--v_reset", type=float, default=0.0, help="reset init (<=0: use v_th)")
    p.add_argument("--v_pre", type=float, default=1.0)

    p.add_argument("--check_every", type=int, default=1)
    p.add_argument("--max_eval_batches", type=int, default=0, help="limit eval batches (<=0: no limit)")

    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--download", type=int, default=0, choices=[0, 1], help="download datasets if missing (0/1)")
    p.add_argument("--gpu", type=int, default=0, help="CUDA GPU index")

    p.add_argument("--lambda_ortho", type=float, default=0.0)
    p.add_argument("--lambda_s", type=float, default=0.0)

    args = p.parse_args()

    if len(args.models) == 1 and str(args.models[0]).strip().lower() == "all":
        models = ALL_MODELS
    else:
        models = [str(m).strip() for m in args.models if str(m).strip()]

    run_acc_benchmark(
        dataset="s-cifar10",
        out_root=args.out_root,
        exp_name=args.exp_name,
        timestamp=args.timestamp,
        data_root=args.data_root,
        hidden_dims=args.hidden,
        models=models,
        epochs=args.epochs,
        soft_mask_epochs=args.soft_mask_epochs,
        stabilize_epochs=args.stabilize_epochs,
        ste_epochs=args.ste_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        weight_decay_dend_soma=args.weight_decay_dend_soma,
        seed=args.seed,
        S_min=args.S_min,
        S_max_list=args.S_max,
        th_len=args.th_len,
        v_th=args.v_th,
        v_reset=None if args.v_reset <= 0 else args.v_reset,
        v_pre=args.v_pre,
        cifar_mode=args.cifar_mode,
        check_every=args.check_every,
        max_eval_batches=None if args.max_eval_batches <= 0 else args.max_eval_batches,
        num_workers=args.num_workers,
        download=bool(args.download),
        device=f"cuda:{args.gpu}",
        lambda_ortho=args.lambda_ortho,
        lambda_s=args.lambda_s,
    )


if __name__ == "__main__":
    main()
