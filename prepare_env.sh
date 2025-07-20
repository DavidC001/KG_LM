#!/bin/bash

source ~/.bashrc

conda activate KG_LFM

module load cuda

export HF_HUB_OFFLINE=1
export WANDB_MODE=offline
