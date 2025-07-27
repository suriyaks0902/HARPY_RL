#!/bin/bash

# Default log directory
LOG_DIR="../logs"

# Check if a specific directory is provided
if [ $# -eq 1 ]; then
    LOG_DIR="$1"
fi

# Launch tensorboard
tensorboard --logdir=${LOG_DIR} --port=6006 --bind_all 