#!/bin/bash
# list all configurations in a given folder and run "sbatch launchers/train.sh conf.yaml"

if [ -z "$1" ]; then
    echo "Please provide a configuration directory."
    exit 1
fi

CONFIG_DIR="$1"

# match anything in the directory that ends with yaml that is not debug.yaml
CONFIG_FILES=$(find "$CONFIG_DIR" -name "*.yaml" ! -name "debug.yaml")

for CONFIG in $CONFIG_FILES; do
    sbatch launchers/train.sh "$CONFIG"
    echo "Submitted job for configuration: $CONFIG"
done
