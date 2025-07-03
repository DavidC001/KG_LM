#!/bin/bash
#SBATCH --job-name=pretrain                # job name
#SBATCH --nodes=1                       # number of nodes
#SBATCH --ntasks-per-node=1             # number of tasks per node
#SBATCH --time=12:00:00                 # time limits
#SBATCH --output=out/pretrain_%j.out                # standard output file
#SBATCH --account=iscrc_kg-lfm        # account name
#SBATCH --partition=boost_usr_prod      # partition name
#SBATCH --gres=gpu:4                    # Generic resources, e.g., GPUs
#SBATCH --cpus-per-task=32
#SBATCH --mem=480GB
#SBATCH --chdir=.                       # start from current directory


export HF_HUB_OFFLINE=1
export WANDB_MODE=offline
module load cuda

source ~/.bashrc
conda activate CF

# Set NCCL environment variables
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800  # 30 minutes
export TORCH_NCCL_ENABLE_MONITORING=1
export NCCL_TIMEOUT=1800
export NCCL_DEBUG=INFO
export NCCL_ASYNC_ERROR_HANDLING=1
export CUDA_LAUNCH_BLOCKING=0

# Optional: Set additional NCCL debugging variables
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL

# Set OMP threads to avoid oversubscription
export OMP_NUM_THREADS=1

echo "Starting training with NCCL timeout settings..."
echo "TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC: $TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC"
echo "NCCL_TIMEOUT: $NCCL_TIMEOUT"


# if no argument is provided, use default config
if [ -z "$1" ]; then
    echo "No config file provided, using default config."
    export CONFIG_FILE="config.yaml"
else
    export CONFIG_FILE=$1
fi

srun accelerate launch train.py --config $CONFIG_FILE
