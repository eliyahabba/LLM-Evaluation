#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Activate the conda environment if it exists
if [ -f "/cs/labs/gabis/eliyahabba/mambaforge/etc/profile.d/conda.sh" ]; then
    source "/cs/labs/gabis/eliyahabba/mambaforge/etc/profile.d/conda.sh"
    conda activate llm_utils
fi

# Change to the script directory
cd "$SCRIPT_DIR"

# Run the dataset merger with lean processing only
python dataset_merger.py --process-type lean 