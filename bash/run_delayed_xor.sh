#!/usr/bin/env bash
set -euo pipefail

# run_delayed_xor.sh : basic_long_term_mem(delayed_XOR) - nohup 1회 실행 (python 내부 직렬)
# stdout: 백그라운드 python PID (숫자 1줄)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# PYTHON : 사용할 python 실행기
PYTHON="${PYTHON:-python3}"

# RUN_PY : 실행할 run.py 경로
RUN_PY="${ROOT_DIR}/src/basic_long_term_mem/delayed_XOR/run.py"

# RESULT_DIR : 실험 결과 저장 루트 (--out_root)
RESULT_DIR="${RESULT_DIR:-${ROOT_DIR}/result}"

mkdir -p "${RESULT_DIR}"

if [[ ! -f "${RUN_PY}" ]]; then
  echo "ERROR: run.py not found: ${RUN_PY}" >&2
  exit 1
fi

# GPU : 사용할 물리 GPU 인덱스 (--gpu)
GPU="${GPU:-0}"

# RUN_PREFIX : 결과 폴더 prefix에 사용할 이름 (--exp_name prefix)
RUN_PREFIX="${RUN_PREFIX:-delayed_xor}"

# SEED : 랜덤 시드 (--seed)
SEED="${SEED:-0}"

# MODELS : 실행할 모델 목록 (--models). 공백 구분
MODELS="${MODELS:-my-dh-snn my-r-dh-snn my-d-rf}"

# EPOCHS : 학습 epoch 수 (--epochs)
EPOCHS="${EPOCHS:-50}"

# SOFT_MASK_EPOCHS / STABILIZE_EPOCHS : 2단계 학습 스케줄
#  - Stage A: 소프트 마스크 학습
#  - Stage B: 소수부 절삭 후 하드 마스크 고정 + 안정화 학습
SOFT_MASK_EPOCHS="${SOFT_MASK_EPOCHS:-${EPOCHS}}"
STABILIZE_EPOCHS="${STABILIZE_EPOCHS:-10}"
STE_EPOCHS="${STE_EPOCHS:-0}"

# STEPS_PER_EPOCH : epoch당 step 수 (--steps_per_epoch)
STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-100}"

# BATCH_SIZE : batch size (--batch_size)
BATCH_SIZE="${BATCH_SIZE:-500}"

# LR : learning rate (--lr)
LR="${LR:-1e-3}"

# WEIGHT_DECAY : AdamW weight decay (layer connection weights only)
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"

# WEIGHT_DECAY_DEND_SOMA : my_R_DH_SNN only (W_mix). If unset, run.py defaults to "follow WEIGHT_DECAY".
WEIGHT_DECAY_DEND_SOMA="${WEIGHT_DECAY_DEND_SOMA:-}"

# CHECK_EVERY : 평가 주기(epoch) (--check_every)
CHECK_EVERY="${CHECK_EVERY:-10}"

# EVAL_BATCHES : 평가 배치 수 (--eval_batches)
EVAL_BATCHES="${EVAL_BATCHES:-20}"

# S_MIN/S_MAX : 가지 수(구조 변수) 제어
#  - 별도의 dendritic 파라미터는 제거됨
S_MIN="${S_MIN:-1.0}"
S_MAX="${S_MAX:-8.2}"

# TH_LEN : adaptive threshold length (--th_len)
TH_LEN="${TH_LEN:-4}"

# V_TH : threshold 초기값 (--v_th)
V_TH="${V_TH:-1.0}"

# V_RESET : reset 초기값 (--v_reset)
#   - V_RESET <= 0 : python에서 v_th로 자동 설정
V_RESET="${V_RESET:-0}"

# V_PRE : V_pre (--v_pre)
V_PRE="${V_PRE:-1.0}"

# LAMBDA_ORTHO : orthogonality regularization 계수 (--lambda_ortho)
LAMBDA_ORTHO="${LAMBDA_ORTHO:-0.1}"

# LAMBDA_S : s regularization 계수 (--lambda_s)
LAMBDA_S="${LAMBDA_S:-0.0001}"

# TIME_STEPS : delayed_XOR time steps (--time_steps)
TIME_STEPS="${TIME_STEPS:-200}"

# CHANNEL_SIZE : delayed_XOR channel size (--channel_size)
CHANNEL_SIZE="${CHANNEL_SIZE:-20}"

# CODING_TIME : delayed_XOR coding time (--coding_time)
CODING_TIME="${CODING_TIME:-10}"

# TEST_TIME : delayed_XOR test time (--test_time)
TEST_TIME="${TEST_TIME:-1}"

# NOISE_RATE : 입력 노이즈 비율 (--noise_rate)
NOISE_RATE="${NOISE_RATE:-0.01}"

# RATE_LOW : 입력 rate 하한 (--rate_low)
RATE_LOW="${RATE_LOW:-0.2}"

# RATE_HIGH : 입력 rate 상한 (--rate_high)
RATE_HIGH="${RATE_HIGH:-0.6}"

read -r -a MODELS_ARR <<< "${MODELS}"

# Timestamp format must match project spec: YYmmdd_HHMMSS
TS="$(date +%y%m%d_%H%M%S)"
MODELS_TAG="${MODELS// /-}"

EXP_NAME="${RUN_PREFIX}_models${MODELS_TAG}_seed${SEED}_Smax${S_MAX}"
EXP_DIR="${RESULT_DIR}/${EXP_NAME}_${TS}"
mkdir -p "${EXP_DIR}"
LOG_FILE="${EXP_DIR}/run.log"

CMD=(
  nohup "${PYTHON}" -u "${RUN_PY}"
    --out_root "${RESULT_DIR}"
    --exp_name "${EXP_NAME}"
    --timestamp "${TS}"
    --gpu "${GPU}"
    --seed "${SEED}"
    --models "${MODELS_ARR[@]}"
    --epochs "${EPOCHS}"
    --soft_mask_epochs "${SOFT_MASK_EPOCHS}"
    --stabilize_epochs "${STABILIZE_EPOCHS}"
    --ste_epochs "${STE_EPOCHS}"
    --steps_per_epoch "${STEPS_PER_EPOCH}"
    --batch_size "${BATCH_SIZE}"
    --lr "${LR}"
    --weight_decay "${WEIGHT_DECAY}"
    --check_every "${CHECK_EVERY}"
    --eval_batches "${EVAL_BATCHES}"
    --S_min "${S_MIN}"
    --S_max "${S_MAX}"
    --th_len "${TH_LEN}"
    --v_th "${V_TH}"
    --v_reset "${V_RESET}"
    --v_pre "${V_PRE}"
    --lambda_ortho "${LAMBDA_ORTHO}"
    --lambda_s "${LAMBDA_S}"
    --time_steps "${TIME_STEPS}"
    --channel_size "${CHANNEL_SIZE}"
    --coding_time "${CODING_TIME}"
    --test_time "${TEST_TIME}"
    --noise_rate "${NOISE_RATE}"
    --rate_low "${RATE_LOW}"
    --rate_high "${RATE_HIGH}"
)

if [[ -n "${WEIGHT_DECAY_DEND_SOMA}" ]]; then
  CMD+=(--weight_decay_dend_soma "${WEIGHT_DECAY_DEND_SOMA}")
fi

"${CMD[@]}" > "${LOG_FILE}" 2>&1 &
pid=$!
echo "${pid}" > "${EXP_DIR}/nohup.pid"

# stdout에는 PID만 출력
echo "${pid}"
