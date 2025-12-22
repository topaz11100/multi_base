#!/bin/bash

# 오류 발생 시 즉시 중단
set -e

SCRIPT_DIR="$(cd -- "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN=${PYTHON:-python}
SHD_DATA_ROOT="${ROOT_DIR}/shd_ssc_data/shd"
SSC_DATA_ROOT="${ROOT_DIR}/shd_ssc_data/ssc"

echo "========================================================"
echo "Starting Quick Check for All Tasks and Models"
echo "========================================================"




# 2. S-MNIST Task
# neurons: cp, tc, ts, dh-sfnn, dh-srnn (Note: S-MNIST uses hyphens)
echo "2. Running S-MNIST Check..."
"${PYTHON_BIN}" task/s-mnist/run.py \
    --exp-name check_smnist \
    --epochs 1 \
    --batch-size 2048 \
    --neurons tc,ts,dh-sfnn,dh-srnn


# 3. Delayed XOR Task
# models: all (includes dh, cp, tc, ts)
# delay_T=0 will run just one delay setting (0)
echo "3. Running Delayed XOR Check..."
"${PYTHON_BIN}" task/delayed_xor/run.py \
    --model all \
    --epochs 1 \
    --batch_size 4 \
    --steps_per_epoch 1 \
    --eval_steps 1 \
    --delay_T 0 \
    --delay_T_delta 10 \
    --hidden_dim 4 


# 4. Multiscale XOR Task
# neurons: dh tc ts cp
# Runs for a single delay configuration (min=10, max=10)
echo "4. Running Multiscale XOR Check..."
"${PYTHON_BIN}" task/multiscale_xor/run.py \
    --exp_name check_multiscale \
    --epochs 1 \
    --batch_size 4 \
    --trials 1 \
    --log_interval 1 \
    --eval_steps 1 \
    --delay_min 10 \
    --delay_max 10 \
    --delay_step 10 \
    --neurons dh tc ts cp

# 1. SHD Task
# models: cp, tc, ts, dh_sfnn, dh_srnn (Note: SHD uses underscores)
echo "1. Running SHD Check..."
"${PYTHON_BIN}" task/SHD/run.py \
    --exp_name check_shd \
    --epochs 1 \
    --batch_size 4 \
    --models cp tc ts dh_sfnn dh_srnn \
    --data_root "${SHD_DATA_ROOT}"

# 5. SSC Task
# neurons: cp, tc, ts, dh-sfnn, dh-srnn (Note: SSC uses hyphens)
echo "5. Running SSC Check..."
"${PYTHON_BIN}" task/SSC/run.py \
    --exp-name check_ssc \
    --epochs 1 \
    --batch-size 4 \
    --neurons cp,tc,ts,dh-sfnn,dh-srnn \
    --data-root "${SSC_DATA_ROOT}"


echo "========================================================"
echo "All checks passed successfully!"
echo "========================================================"
