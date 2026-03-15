#!/usr/bin/env bash
set -euo pipefail

# run_freq_shd.sh : freq_analysis(SHD) - nohup 1회 실행 (python 내부에서 모델들 직렬 실행)
# stdout: 백그라운드 python PID (숫자 1줄)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON="${PYTHON:-python3}"
RUN_PY="${ROOT_DIR}/src/freq_analysis/SHD/run.py"

RESULT_DIR="${RESULT_DIR:-${ROOT_DIR}/result}"
DATA_DIR="${DATA_DIR:-${ROOT_DIR}/data}"

mkdir -p "${RESULT_DIR}"

if [[ ! -f "${RUN_PY}" ]]; then
  echo "ERROR: run.py not found: ${RUN_PY}" >&2
  exit 1
fi

GPU="${GPU:-0}"
RUN_PREFIX="${RUN_PREFIX:-freq_shd}"
SEED="${SEED:-0}"

MODELS="${MODELS:-my_R_DH_SNN}"

HIDDEN="${HIDDEN:-32 32 32}"
EPOCHS="${EPOCHS:-90}"

# 2-stage schedule for variable-branch models
# - SOFT_MASK_EPOCHS: Stage A length (soft mask; optionally STE at end)
# - STE_EPOCHS: last STE_EPOCHS epochs of Stage A use STE (forward hard / backward soft)
# - STABILIZE_EPOCHS: Stage B length after hardening (0 disables harden/Stage B)
SOFT_MASK_EPOCHS="${SOFT_MASK_EPOCHS:-${EPOCHS}}"
# NOTE: run must be robust under `set -u`; provide a default for STE_EPOCHS.
STE_EPOCHS="${STE_EPOCHS:-10}"
STABILIZE_EPOCHS="${STABILIZE_EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
LR="${LR:-1e-2}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
# Optional: only used by my_R_DH_SNN (W_mix). If unset, run.py defaults to "follow WEIGHT_DECAY".
WEIGHT_DECAY_DEND_SOMA="${WEIGHT_DECAY_DEND_SOMA:-}"

T_EVENT="${T_EVENT:-250}"

S_MIN="${S_MIN:-1.0}"
S_MAX="${S_MAX:-8.2}"

## Readout is fixed to membrane potential (mem).

TH_LEN="${TH_LEN:-4}"
V_TH="${V_TH:-1.0}"
V_PRE="${V_PRE:-1.0}"

# -----------------------------
# Analysis / logging hyperparams
# -----------------------------
# PLOT_EVERY      : model-level distributions (timing/structure/weights)
# ANALYSIS_EVERY  : per-label probe analysis + per-neuron signal/Δ plots
# CONV_EVERY      : (enable/disable) convergence Δ plots (saved at final epoch only)
# ANALYSIS_NEURONS: per-hidden-layer *sample counts* (e.g., "5 5 5").
#                  Sampled indices are recorded in config.json.
PLOT_EVERY="${PLOT_EVERY:-10}"
ANALYSIS_EVERY="${ANALYSIS_EVERY:-10}"
CONV_EVERY="${CONV_EVERY:-10}"

# Per-hidden-layer sample counts (default: 5 per hidden layer)
ANALYSIS_NEURONS="${ANALYSIS_NEURONS:-3 3 3}"
FFT_BAND_EDGES="${FFT_BAND_EDGES:-0.0 0.025 0.05 0.075 0.1 0.125 0.15 0.175 0.2 0.225 0.25 0.275 0.3 0.325 0.35 0.375 0.4 0.425 0.45 0.475 0.5}"
FFT_BAND_REDUCE="${FFT_BAND_REDUCE:-mean}"

NUM_WORKERS="${NUM_WORKERS:-4}"
DOWNLOAD="${DOWNLOAD:-0}"

LAMBDA_ORTHO="${LAMBDA_ORTHO:-0.1}"
LAMBDA_S="${LAMBDA_S:-0.0001}"

read -r -a MODELS_ARR <<< "${MODELS}"
read -r -a HIDDEN_ARR <<< "${HIDDEN}"

if [[ -z "${ANALYSIS_NEURONS}" ]]; then
  ANALYSIS_NEURONS=""
  for _ in "${HIDDEN_ARR[@]}"; do
    ANALYSIS_NEURONS+="5 "
  done
  ANALYSIS_NEURONS="${ANALYSIS_NEURONS%% }"
fi
read -r -a ANALYSIS_NEURONS_ARR <<< "${ANALYSIS_NEURONS}"
read -r -a FFT_BAND_EDGES_ARR <<< "${FFT_BAND_EDGES}"

TS="$(date +%y%m%d_%H%M%S)"
EXP_PREFIX="${RUN_PREFIX}_seed${SEED}_Smax${S_MAX}_Tev${T_EVENT}"

LOG_FILE="${RESULT_DIR}/${EXP_PREFIX}_${TS}.log"
PID_FILE="${RESULT_DIR}/${EXP_PREFIX}_${TS}.pid"

CMD=(
  nohup "${PYTHON}" -u "${RUN_PY}"
    --out_root "${RESULT_DIR}"
    --data_root "${DATA_DIR}"
    --exp_name "${EXP_PREFIX}"
    --timestamp "${TS}"
    --gpu "${GPU}"
    --seed "${SEED}"
    --models "${MODELS_ARR[@]}"
    --hidden "${HIDDEN_ARR[@]}"
    --epochs "${EPOCHS}"
    --soft_mask_epochs "${SOFT_MASK_EPOCHS}"
    --stabilize_epochs "${STABILIZE_EPOCHS}"
    --ste_epochs "${STE_EPOCHS}"
    --batch_size "${BATCH_SIZE}"
    --lr "${LR}"
    --weight_decay "${WEIGHT_DECAY}"
    --T_event "${T_EVENT}"
    --S_min "${S_MIN}"
    --S_max "${S_MAX}"
    --th_len "${TH_LEN}"
    --v_th "${V_TH}"
    --v_pre "${V_PRE}"
    --plot_every "${PLOT_EVERY}"
    --analysis_every "${ANALYSIS_EVERY}"
    --convergence_every "${CONV_EVERY}"
    --fft_band_reduce "${FFT_BAND_REDUCE}"
    --num_workers "${NUM_WORKERS}"
    --download "${DOWNLOAD}"
    --lambda_ortho "${LAMBDA_ORTHO}"
    --lambda_s "${LAMBDA_S}"
)

if [[ -n "${WEIGHT_DECAY_DEND_SOMA}" ]]; then
  CMD+=(--weight_decay_dend_soma "${WEIGHT_DECAY_DEND_SOMA}")
fi

if [[ ${#ANALYSIS_NEURONS_ARR[@]} -gt 0 ]]; then
  CMD+=(--analysis_neurons "${ANALYSIS_NEURONS_ARR[@]}")
fi

if [[ ${#FFT_BAND_EDGES_ARR[@]} -gt 0 ]]; then
  CMD+=(--fft_band_edges "${FFT_BAND_EDGES_ARR[@]}")
fi

"${CMD[@]}" > "${LOG_FILE}" 2>&1 &
pid=$!
echo "${pid}" > "${PID_FILE}"

echo "${pid}"
