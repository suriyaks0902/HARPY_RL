# export CUDA_VISIBLE_DEVICES=""
# mpirun -np 1 python ../scripts/train.py  --train_name 'vnav_phase_thruster_v1' \
#                                         --rnd_seed 1 \
#                                         --max_iters 6000 \
#                                         --save_interval 100 \
#                                         #--restore_from 'previous_ckpts'

#!/bin/bash

# GPU Configuration
export CUDA_VISIBLE_DEVICES="0"  # Use first GPU, set to "" for CPU only

# Environment variables for MuJoCo
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6

# Training Configuration
TRAIN_NAME="cassie_jump_training"
MAX_ITERS_PER_STAGE=6000
SAVE_INTERVAL=100
RND_SEED=42

# Function to check if previous command was successful
check_status() {
    if [ $? -ne 0 ]; then
        echo "Error: Stage $1 training failed!"
        exit 1
    fi
}

# Create necessary directories
mkdir -p ../logs/${TRAIN_NAME}
mkdir -p ../ckpts/${TRAIN_NAME}

# Set environment variables for logging
export OPENAI_LOG_FORMAT=stdout,log,csv,tensorboard
export OPENAI_LOGDIR=../logs/${TRAIN_NAME}

# Choose training mode (uncomment one section)

### Option 1: Full Curriculum Training ###
echo "Starting full curriculum training..."
mpirun -np 1 python ../scripts/train.py \
    --train_name ${TRAIN_NAME} \
    --curriculum \
    --max_iters ${MAX_ITERS_PER_STAGE} \
    --save_interval ${SAVE_INTERVAL} \
    --rnd_seed ${RND_SEED} \
    --gpu 0 \
    # --env_config "{'step_zerotorque': False, 'is_visual': False, 'minimal_rand': True, 'is_noisy': True, 'add_perturbation': False, 'add_standing': True, 'fixed_gait': True, 'add_rotation': False, 'cam_track_robot': False}"

### Option 2: Individual Stage Training ###
# echo "Starting Stage 1 (Stabilization)..."
# mpirun -np 1 python ../scripts/train.py \
#     --train_name ${TRAIN_NAME} \
#     --stage 1 \
#     --max_iters ${MAX_ITERS_PER_STAGE} \
#     --save_interval ${SAVE_INTERVAL} \
#     --rnd_seed ${RND_SEED} \
#     --gpu 0
# check_status 1

# echo "Starting Stage 2 (Jumping)..."
# mpirun -np 1 python ../scripts/train.py \
#     --train_name ${TRAIN_NAME} \
#     --stage 2 \
#     --max_iters ${MAX_ITERS_PER_STAGE} \
#     --save_interval ${SAVE_INTERVAL} \
#     --rnd_seed ${RND_SEED} \
#     --gpu 0
# check_status 2

# echo "Starting Stage 3 (Recovery)..."
# mpirun -np 1 python ../scripts/train.py \
#     --train_name ${TRAIN_NAME} \
#     --stage 3 \
#     --max_iters ${MAX_ITERS_PER_STAGE} \
#     --save_interval ${SAVE_INTERVAL} \
#     --rnd_seed ${RND_SEED} \
#     --gpu 0
# check_status 3

### Option 3: Resume Training from Checkpoint ###
# CHECKPOINT_PATH="ckpts/${TRAIN_NAME}_stage1/model_final"
# echo "Resuming training from checkpoint: ${CHECKPOINT_PATH}"
# mpirun -np 1 python ../scripts/train.py \
#     --train_name ${TRAIN_NAME} \
#     --restore_from ${CHECKPOINT_PATH} \
#     --restore_cont 1 \
#     --max_iters ${MAX_ITERS_PER_STAGE} \
#     --save_interval ${SAVE_INTERVAL} \
#     --rnd_seed ${RND_SEED} \
#     --gpu 0

### Option 4: Regular Training (No Curriculum) ###
# echo "Starting regular training..."
# mpirun -np 1 python ../scripts/train.py \
#     --train_name ${TRAIN_NAME} \
#     --max_iters ${MAX_ITERS_PER_STAGE} \
#     --save_interval ${SAVE_INTERVAL} \
#     --rnd_seed ${RND_SEED} \
#     --gpu 0
check_status "curriculum"

echo "Training complete!"