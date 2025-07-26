#!/bin/bash
#SBATCH --nodes=5                       # number of nodes

#SBATCH --job-name=ray_sweep            # job name
#SBATCH --time=24:00:00               # time limits
#SBATCH --output=out/ray_sweep_%j.out   # standard output file

#SBATCH --chdir=.                       # start from current directory
#SBATCH --ntasks-per-node=1             # number of tasks per node
#SBATCH --account=iscrc_kg-lfm          # account name
#SBATCH --partition=boost_usr_prod      # partition name
#SBATCH --qos=normal

#SBATCH --gpus-per-task=4                 
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=15GB

#SBATCH --mail-type=END,FAIL            # email notification on job end or failure
#SBATCH --mail-user=davide.cavicchini@studenti.unitn.it

source ./prepare_env.sh

# Parse command line arguments
base_conf=$1
if [ -z "$base_conf" ]; then
    echo "No base config file provided, using default config."
    # get pwd and append the default base config file
    pwd=$(pwd)
    echo "Current working directory: $pwd"
    base_conf="$pwd/configs/sweep_base_config.yaml"
else
    echo "Using base config file: $base_conf"
fi


time_budget=$(( 23 * 3600 )) # Default to 23 hours in seconds
# if a $2 argument is provided, use it as time budget
if [ ! -z "$2" ]; then
    time_budget=$2
    echo "Using provided time budget: $time_budget seconds"
else
    echo "No time budget provided, using default: $time_budget seconds"
fi


# Get the list of allocated nodes
all_nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST)
head_node=$(echo $all_nodes | awk '{print $1}')  # First node is head node

echo "Head node: $head_node"

# Get head node IP
head_node_ip=$(srun --nodes=1 --ntasks=1 -w $head_node hostname --ip-address)
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
echo "All allocated nodes: $all_nodes"

# Start the head node
srun --nodes=1 --ntasks=1 -w $head_node launchers/ray/start-head.sh $head_node_ip &

# Wait for the head node to start
sleep 10

worker_num=$(($SLURM_JOB_NUM_NODES - 1)) #number of nodes other than the head node

# Start worker nodes
if [ $worker_num -gt 0 ]; then
    srun -n $worker_num --nodes=$worker_num --ntasks-per-node=1 --exclude $head_node launchers/ray/start-worker.sh $head_node_ip:$port &
fi
# Wait for all workers to start
sleep 5
##############################################################################################

#### call your code below
python sweep.py \
    --base_conf $base_conf \
    --time_budget $time_budget
