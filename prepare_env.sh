#!/bin/bash

conda activate CF

module load cuda

export HF_HUB_OFFLINE=1
export WANDB_MODE=offline

# Set NCCL environment variables for better timeout and error handling
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600  # 60 minutes
export TORCH_NCCL_ENABLE_MONITORING=1
export NCCL_TIMEOUT=3600  # 60 minutes in seconds
export NCCL_ASYNC_ERROR_HANDLING=1
export CUDA_LAUNCH_BLOCKING=0

# Optional: Set additional NCCL debugging variables
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL

# Set OMP threads to avoid oversubscription
export OMP_NUM_THREADS=1