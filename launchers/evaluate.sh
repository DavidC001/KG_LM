#!/bin/bash
#SBATCH --job-name=eval                # job name
#SBATCH --nodes=1                       # number of nodes
#SBATCH --ntasks-per-node=1             # number of tasks per node
#SBATCH --time=1-00:00:00                 # time limits
#SBATCH --output=out/eval_%j.out                # standard output file
#SBATCH --account=iscrc_kg-lfm        # account name
#SBATCH --partition=boost_usr_prod      # partition name
#SBATCH --gres=gpu:1                    # Generic resources, e.g., GPUs
#SBATCH --cpus-per-task=8
#SBATCH --mem=120GB
#SBATCH --chdir=.                       # start from current directory
#SBATCH --mail-type=END,FAIL            # email notification on job end or failure
#SBATCH --mail-user=davide.cavicchini@studenti.unitn.it

source ./prepare_env.sh

# if no argument is provided, use default config
if [ -z "$1" ]; then
    echo "No config file provided, using default config."
    export CONFIG_FILE="configs/pretrain_config.yaml"
else
    export CONFIG_FILE=$1
fi

if [ -z "$2" ]; then
    echo "No batch size provided, using default batch size of 16."
    export BATCH_SIZE=16
else
    export BATCH_SIZE=$2
fi

if [ -z "$3" ]; then
    echo "No max samples provided, using default max samples of None."
    export MAX_SAMPLES="None"
else
    export MAX_SAMPLES=$3
fi

python evaluate.py --config $CONFIG_FILE --batch_size $BATCH_SIZE --max_samples $MAX_SAMPLES
