#!/bin/bash

# 오류 발생 시 즉시 중단
set -e

echo "========================================================"
echo "Starting Quick Check for All Tasks and Models"
echo "========================================================"

# 1. SHD Task
# models: cp, tc, ts, dh_sfnn, dh_srnn (Note: SHD uses underscores)
echo "1. Running SHD Check..."
python task/SHD/run.py \
    --exp_name check_shd \
    --epochs 1 \
    --batch_size 4 \
    --models cp tc ts dh_sfnn dh_srnn \
    

# 2. S-MNIST Task
# neurons: cp, tc, ts, dh-sfnn, dh-srnn (Note: S-MNIST uses hyphens)
echo "2. Running S-MNIST Check..."
python task/s-mnist/run.py \
    --exp-name check_smnist \
    --epochs 1 \
    --batch-size 4 \
    --neurons cp,tc,ts,dh-sfnn,dh-srnn \
    

# 3. Delayed XOR Task
# models: all (includes dh, cp, tc, ts)
# delay_T=0 will run just one delay setting (0)
echo "3. Running Delayed XOR Check..."
python task/delayed_xor/run.py \
    --model all \
    --epochs 1 \
    --batch_size 4 \
    --steps_per_epoch 1 \
    --eval_steps 1 \
    --delay_T 0 \
    --delay_T_delta 10 \
    --hidden_dim 4 \
    

# 4. Multiscale XOR Task
# neurons: dh tc ts cp
# Runs for a single delay configuration (min=10, max=10)
echo "4. Running Multiscale XOR Check..."
python task/multiscale_xor/run.py \
    --exp_name check_multiscale \
    --epochs 1 \
    --batch_size 4 \
    --trials 1 \
    --log_interval 1 \
    --eval_steps 1 \
    --delay_min 10 \
    --delay_max 10 \
    --delay_step 10 \
    --neurons dh tc ts cp \
    

# 5. SSC Task
# neurons: cp, tc, ts, dh-sfnn, dh-srnn (Note: SSC uses hyphens)
echo "5. Running SSC Check..."
python task/SSC/run.py \
    --exp-name check_ssc \
    --epochs 1 \
    --batch-size 4 \
    --neurons cp,tc,ts,dh-sfnn,dh-srnn \
    

echo "========================================================"
echo "All checks passed successfully!"
echo "========================================================"