#!/bin/bash
#SBATCH --job-name=pretrain                # job name
#SBATCH --nodes=1                       # number of nodes
#SBATCH --ntasks-per-node=1             # number of tasks per node
#SBATCH --time=1-00:00:00                 # time limits
#SBATCH --output=out/pretrain_%j.out                # standard output file
#SBATCH --account=iscrc_kg-lfm        # account name
#SBATCH --partition=boost_usr_prod      # partition name
#SBATCH --gres=gpu:4                    # Generic resources, e.g., GPUs
#SBATCH --cpus-per-task=32
#SBATCH --mem=480GB
#SBATCH --chdir=.                       # start from current directory
#SBATCH --mail-type=END,FAIL            # email notification on job end or failure
#SBATCH --mail-user=davide.cavicchini@studenti.unitn.it

source ./prepare_env.sh

echo "Starting training with enhanced NCCL timeout and DeepSpeed settings..."
echo "TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC: $TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC"
echo "NCCL_TIMEOUT: $NCCL_TIMEOUT"

# if no argument is provided, use default config
if [ -z "$1" ]; then
    echo "No config file provided, using default config."
    export CONFIG_FILE="configs/pretrain_config.yaml"
else
    export CONFIG_FILE=$1
fi

# Use accelerate launch with explicit deepspeed config
srun accelerate launch \
    --config_file configs/accelerate_config.yaml \
    train.py --config $CONFIG_FILE
