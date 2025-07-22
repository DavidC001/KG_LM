#!/bin/bash

source ~/.bashrc

conda activate KG_LFM

module unload cuda
module load cuda/12.1

export HF_HUB_OFFLINE=1
export WANDB_MODE=offline
