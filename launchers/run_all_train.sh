#!/bin/bash
# list all configurations in a given folder and run "sbatch launchers/train.sh conf.yaml"

if [ -z "$1" ]; then
    echo "Please provide a configuration directory."
    exit 1
fi

CONFIG_DIR="$1"

if [ -z "$2" ]; then
    TIME_BUDGET=$((3600 * 24 - 60 * 30))
else
    TIME_BUDGET="$2"
fi

# match anything in the directory that ends with yaml that is not debug.yaml
CONFIG_FILES=$(find "$CONFIG_DIR" -name "*.yaml" ! -name "debug.yaml")

for CONFIG in $CONFIG_FILES; do
    sbatch launchers/train.sh "$CONFIG" "$TIME_BUDGET"
    echo "Submitted job for configuration: $CONFIG with time budget: $TIME_BUDGET"
done
