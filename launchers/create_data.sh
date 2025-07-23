#!/bin/bash
#SBATCH --job-name=create_data                # job name
#SBATCH --nodes=1                       # number of nodes
#SBATCH --ntasks-per-node=1             # number of tasks per node
#SBATCH --time=12:00:00                 # time limits
#SBATCH --output=out/create_data_%j.out                # standard output file
#SBATCH --account=iscrc_kg-lfm        # account name
#SBATCH --partition=boost_usr_prod      # partition name
#SBATCH --gres=gpu:0                    # Generic resources, e.g., GPUs
#SBATCH --cpus-per-task=16
#SBATCH --mem=256GB
#SBATCH --chdir=.                       # start from current directory


export HF_HUB_OFFLINE=1
export WANDB_MODE=offline

source ~/.bashrc
conda activate CF

# if no argument is provided, use default config
if [ -z "$1" ]; then
    echo "No config file provided, using default config."
    export CONFIG_FILE="configs/pretrain_config.yaml"
else
    export CONFIG_FILE=$1
fi

python create_hf_dataset.py --config $CONFIG_FILE
