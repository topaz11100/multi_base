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
    --epochs 25 \
    --batch-size 1024 \
    --neurons ts

"${PYTHON_BIN}" task/s-mnist/run.py \
    --exp-name check_smnist \
    --epochs 25 \
    --batch-size 2048 \
    --neurons dh-srnn,dh-sfnn,tc,cp



echo "========================================================"
echo "All checks passed successfully!"
echo "========================================================"
