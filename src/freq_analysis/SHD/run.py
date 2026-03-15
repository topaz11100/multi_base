import argparse
import os
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_ROOT = os.path.abspath(os.path.join(THIS_DIR, "../../.."))
sys.path.insert(0, PROJ_ROOT)

from src.common.freq_driver import run_freq_analysis  # noqa: E402
from src.common.utils import now_timestamp_seoul  # noqa: E402


ALLOWED_MODELS = ["my_DH_SNN", "my_R_DH_SNN", "my_D_RF"]


def main():
    p = argparse.ArgumentParser(description="Frequency analysis experiment: SHD")

    p.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=ALLOWED_MODELS,
        choices=ALLOWED_MODELS,
        help="one or more models: my_DH_SNN | my_R_DH_SNN | my_D_RF",
    )

    p.add_argument("--out_root", type=str, default=os.path.join(PROJ_ROOT, "result"))
    p.add_argument("--data_root", type=str, default=os.path.join(PROJ_ROOT, "data"))
    p.add_argument("--hidden", type=int, nargs="+", default=[256], help="hidden layer sizes (e.g., --hidden 256 256)")
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
    p.add_argument(
        "--timestamp",
        type=str,
        default=None,
        help="override YYmmdd_HHMMSS timestamp (kept identical across all models in this run)",
    )

    p.add_argument("--T_event", type=int, default=250)

    # Branch count is controlled ONLY by S_min/S_max (no separate dendritic arg)
    p.add_argument("--S_min", type=float, default=1.0, help="lower bound for continuous structural param s")
    p.add_argument("--S_max", type=float, default=8.0, help="upper bound for continuous s (also sets max branch dimension)")

    # Readout is fixed to membrane potential (mem).

    p.add_argument("--th_len", type=int, default=4)
    p.add_argument(
        "--v_th",
        type=float,
        default=1.0,
        help="spike threshold (used in DH/R-DH; D_RF uses adaptive threshold)",
    )
    p.add_argument("--v_pre", type=float, default=1.0, help="V_pre for D_RF adaptive threshold")

    p.add_argument("--plot_every", type=int, default=5)
    p.add_argument("--analysis_every", type=int, default=5)
    p.add_argument("--convergence_every", type=int, default=5)
    p.add_argument(
        "--analysis_neurons",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Per-hidden-layer sample counts for detailed neuron analysis (length must match --hidden). "
            "Example: '--analysis_neurons 5 5 5' samples 5 random neurons from each hidden layer. "
            "Sampled indices are recorded in config.json."
        ),
    )

    p.add_argument(
        "--fft_band_edges",
        type=float,
        nargs="+",
        default=None,
        help="FFT band edges in cycles/step: e0 e1 ... eB",
    )
    p.add_argument("--fft_band_reduce", type=str, default="mean", choices=["mean", "sum", "l2", "max"])

    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--download", type=int, default=0, choices=[0, 1], help="download datasets if missing (0/1)")
    p.add_argument("--gpu", type=int, default=0, help="CUDA GPU index")

    p.add_argument("--lambda_ortho", type=float, default=0.0)
    p.add_argument("--lambda_s", type=float, default=0.0)

    args = p.parse_args()

    ts = args.timestamp or now_timestamp_seoul()
    models = list(args.models)

    for m in models:
        exp_name_m = args.exp_name
        if exp_name_m is not None and len(models) > 1:
            exp_name_m = f"{exp_name_m}_{m}"

        run_freq_analysis(
            dataset="SHD",
            model=m,
            out_root=args.out_root,
            exp_name=exp_name_m,
            timestamp=ts,
            data_root=args.data_root,
            hidden=args.hidden,
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
            S_max=args.S_max,
            th_len=args.th_len,
            v_th=args.v_th,
            v_pre=args.v_pre,
            plot_every=args.plot_every,
            analysis_every=args.analysis_every,
            convergence_every=args.convergence_every,
            analysis_neurons=args.analysis_neurons,
            fft_band_edges=args.fft_band_edges,
            fft_band_reduce=args.fft_band_reduce,
            num_workers=args.num_workers,
            download=bool(args.download),
            T_event=args.T_event,
            lambda_ortho=args.lambda_ortho,
            lambda_s=args.lambda_s,
            device=f"cuda:{args.gpu}",
        )


if __name__ == "__main__":
    main()
