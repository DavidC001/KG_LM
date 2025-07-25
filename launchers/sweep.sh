#!/bin/bash
#SBATCH --nodes=10                       # number of nodes

#SBATCH --job-name=ray_sweep            # job name
#SBATCH --time=1-00:00:00               # time limits
#SBATCH --output=out/ray_sweep_%j.out   # standard output file

#SBATCH --ntasks-per-node=1             # number of tasks per node
#SBATCH --account=iscrc_kg-lfm          # account name
#SBATCH --partition=boost_usr_prod      # partition name

#SBATCH --gpus-per-task=4                 
#SBATCH --cpus-per-task=32
#SBATCH --mem=480GB

#SBATCH --mail-type=END,FAIL            # email notification on job end or failure
#SBATCH --mail-user=davide.cavicchini@studenti.unitn.it

source ./prepare_env.sh

# Parse command line arguments
base_conf=$1
num_samples=${2:-50}
max_concurrent_trials=${3:-4}

if [ -z "$base_conf" ]; then
    echo "No base config file provided, using default config."
    base_conf="configs/sweep_base_config.yaml"
else
    echo "Using base config file: $base_conf"
fi

head_node=$(hostname)
head_node_ip=$(hostname --ip-address)
# if we detect a space character in the head node IP, we'll
# convert it to an ipv4 address. This step is optional.
if [[ "$head_node_ip" == *" "* ]]; then
    IFS=' ' read -ra ADDR <<<"$head_node_ip"
    if [[ ${#ADDR[0]} -gt 16 ]]; then
        head_node_ip=${ADDR[1]}
    else
        head_node_ip=${ADDR[0]}
    fi
fi
port=6379

##############################################################################################
echo "STARTING HEAD at $head_node"
echo "Head node IP: $head_node_ip"

# Start the head node
srun --nodes=1 --ntasks=1 -w $head_node start-head.sh $head_node_ip &
# Wait for the head node to start
sleep 10

worker_num=$(($SLURM_JOB_NUM_NODES - 1)) #number of nodes other than the head node

# Start worker nodes
srun -n $worker_num --nodes=$worker_num --ntasks-per-node=1 --exclude $head_node start-worker.sh $head_node_ip:$port &
# Wait for all workers to start
sleep 5
##############################################################################################

#### call your code below
python sweep.py --base_config $base_conf

exit