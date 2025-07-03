#!/bin/bash

conda activate CF

module load cuda

export HF_HUB_OFFLINE=1
export WANDB_MODE=offline