#!/usr/bin/env bash
set -euo pipefail

# run_acc_smnist.sh : acc_benchmark(sMNIST) - nohup 1회 실행
# stdout: 백그라운드 python PID (숫자 1줄)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON="${PYTHON:-python3}"
RUN_PY="${ROOT_DIR}/src/acc_benchmark/s-mnist/run.py"

RESULT_DIR="${RESULT_DIR:-${ROOT_DIR}/result}"
DATA_DIR="${DATA_DIR:-${ROOT_DIR}/data}"

mkdir -p "${RESULT_DIR}"

if [[ ! -f "${RUN_PY}" ]]; then
  echo "ERROR: run.py not found: ${RUN_PY}" >&2
  exit 1
fi

GPU="${GPU:-0}"
RUN_PREFIX="${RUN_PREFIX:-acc_smnist}"
SEED="${SEED:-0}"

MODELS_RAW="${MODELS:-all}"
MODELS_RAW="${MODELS_RAW//,/ }"
HIDDEN="${HIDDEN:-128}"
EPOCHS="${EPOCHS:-50}"

# 2-stage schedule for variable-branch models
# - SOFT_MASK_EPOCHS: Stage A length (soft mask; optionally STE at end)
# - STE_EPOCHS: last STE_EPOCHS epochs of Stage A use STE (forward hard / backward soft)
# - STABILIZE_EPOCHS: Stage B length after hardening (0 disables harden/Stage B)
SOFT_MASK_EPOCHS="${SOFT_MASK_EPOCHS:-${EPOCHS}}"
STABILIZE_EPOCHS="${STABILIZE_EPOCHS:-0}"
STE_EPOCHS="${STE_EPOCHS:-0}"
BATCH_SIZE="${BATCH_SIZE:-128}"
LR="${LR:-1e-3}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
# Optional: only used by my_R_DH_SNN (W_mix). If unset, run.py defaults to "follow WEIGHT_DECAY".
WEIGHT_DECAY_DEND_SOMA="${WEIGHT_DECAY_DEND_SOMA:-}"

## Readout is fixed to membrane potential (mem).

CHECK_EVERY="${CHECK_EVERY:-1}"
MAX_EVAL_BATCHES="${MAX_EVAL_BATCHES:-0}"

# Branch count control: pass one or more S_max values
S_MAX_LIST="${S_MAX_LIST:-8.0}"
S_MIN="${S_MIN:-1.0}"

TH_LEN="${TH_LEN:-4}"
V_TH="${V_TH:-1.0}"
V_RESET="${V_RESET:-0}"
V_PRE="${V_PRE:-1.0}"

LAMBDA_ORTHO="${LAMBDA_ORTHO:-0.0}"
LAMBDA_S="${LAMBDA_S:-0.0}"

NUM_WORKERS="${NUM_WORKERS:-4}"
DOWNLOAD="${DOWNLOAD:-0}"

read -r -a HIDDEN_ARR <<< "${HIDDEN}"
read -r -a S_MAX_ARR <<< "${S_MAX_LIST}"
read -r -a MODELS_ARR <<< "${MODELS_RAW}"

TS="$(date +%y%m%d_%H%M%S)"
MODELS_TAG="$(IFS='-'; echo "${MODELS_ARR[*]}")"
S_MAX_TAG="$(IFS='-'; echo "${S_MAX_ARR[*]}")"

EXP_NAME="${RUN_PREFIX}_sMNIST_models${MODELS_TAG}_Smax${S_MAX_TAG}_seed${SEED}"
EXP_DIR="${RESULT_DIR}/${EXP_NAME}_${TS}"
mkdir -p "${EXP_DIR}"
LOG_FILE="${EXP_DIR}/run.log"

CMD=(
  nohup "${PYTHON}" -u "${RUN_PY}"
    --out_root "${RESULT_DIR}"
    --data_root "${DATA_DIR}"
    --exp_name "${EXP_NAME}"
    --timestamp "${TS}"
    --models "${MODELS_ARR[@]}"
    --hidden "${HIDDEN_ARR[@]}"
    --epochs "${EPOCHS}"
    --soft_mask_epochs "${SOFT_MASK_EPOCHS}"
    --stabilize_epochs "${STABILIZE_EPOCHS}"
    --ste_epochs "${STE_EPOCHS}"
    --batch_size "${BATCH_SIZE}"
    --lr "${LR}"
    --weight_decay "${WEIGHT_DECAY}"
    --seed "${SEED}"
    --gpu "${GPU}"
    --num_workers "${NUM_WORKERS}"
    --S_min "${S_MIN}"
    --S_max "${S_MAX_ARR[@]}"
    --th_len "${TH_LEN}"
    --v_th "${V_TH}"
    --v_reset "${V_RESET}"
    --v_pre "${V_PRE}"
    --check_every "${CHECK_EVERY}"
    --max_eval_batches "${MAX_EVAL_BATCHES}"
    --lambda_ortho "${LAMBDA_ORTHO}"
    --lambda_s "${LAMBDA_S}"
    --download "${DOWNLOAD}"
)

if [[ -n "${WEIGHT_DECAY_DEND_SOMA}" ]]; then
  CMD+=(--weight_decay_dend_soma "${WEIGHT_DECAY_DEND_SOMA}")
fi

"${CMD[@]}" > "${LOG_FILE}" 2>&1 &
pid=$!
echo "${pid}" > "${EXP_DIR}/nohup.pid"

echo "${pid}"
