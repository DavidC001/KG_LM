#!/bin/bash
#SBATCH --job-name=train             # job name
#SBATCH --nodes=1                       # number of nodes
#SBATCH --time=1-00:00:00               # time limits
#SBATCH --output=out/train_%j.out    # standard output file
#SBATCH --account=iscrc_kg-lfm          # account name
#SBATCH --partition=boost_usr_prod      # partition name
#SBATCH --gpus-per-node=4               # number of GPUs per node
#SBATCH --cpus-per-task=32
#SBATCH --mem=480GB                     # total memory per node
#SBATCH --chdir=.                       # start from current directory
#SBATCH --mail-type=END,FAIL            # email notification on job end or failure
#SBATCH --mail-user=davide.cavicchini@studenti.unitn.it

source ./prepare_env.sh

export TIME_BUDGET=$((3600*24-60*30)) 

echo "Starting training with enhanced NCCL timeout and DeepSpeed settings..."
echo "TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC: $TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC"
echo "NCCL_TIMEOUT: $NCCL_TIMEOUT"

# if no argument is provided, use default config
if [ -z "$1" ]; then
    echo "No config file provided, using default config."
    export CONFIG_FILE="configs/base_config.yaml"
else
    export CONFIG_FILE=$1
    echo "Using provided config file: $CONFIG_FILE"
fi

# Get the list of allocated nodes
all_nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST)
head_node=$(echo $all_nodes | awk '{print $1}')  # First node is head node
head_node_ip=$(srun --nodes=1 --ntasks=1 -w $head_node hostname --ip-address)

# Launch the training script with the specified configuration
export LAUNCHER="accelerate launch --config_file configs/accelerate_config.yaml --main_process_ip $head_node_ip --main_process_port 29500"
export PYTHON_FILE="train.py"
export ARGS="--config $CONFIG_FILE --time_budget $TIME_BUDGET"

export CMD="$LAUNCHER $PYTHON_FILE $ARGS" 
srun $CMD