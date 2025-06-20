#!/bin/bash
#SBATCH --job-name=embed_graph_nodes                # job name
#SBATCH --nodes=1                       # number of nodes
#SBATCH --ntasks-per-node=1             # number of tasks per node
#SBATCH --time=24:00:00                 # time limits
#SBATCH --output=out/embed_graph_%j.out                # standard output file
#SBATCH --account=iscrc_kg-lfm        # account name
#SBATCH --partition=boost_usr_prod      # partition name
#SBATCH --gres=gpu:1                    # Generic resources, e.g., GPUs
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
    export CONFIG_FILE="config.yaml"
else
    export CONFIG_FILE=$1
fi

# second argument is batch size, default is 64
if [ -z "$2" ]; then
    echo "No batch size provided, using default batch size of 64."
    export BATCH_SIZE=64
else
    export BATCH_SIZE=$2
fi

python graph_embedder.py --config $CONFIG_FILE --batch_size $BATCH_SIZE
