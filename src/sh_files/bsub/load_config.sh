#!/bin/bash

# Example of how to use the Python script to get the 'hf_home' configuration
# Assuming the Python script is two levels up from the current directory
config_path="../../../config/get_config.py"

# Ensure the Python script exists and is executable

# Scripts may run from their dir
if [ ! -f "$config_path" ]; then
    config_path="../../../config/get_config.py"
fi
# Ensure the Python script exists and is executable
if [ ! -f "$config_path" ]; then
    config_path="../../../config/get_config.py"
fi

if [ ! -f "$config_path" ]; then
    echo "Configuration script not found at $config_path"
    exit 1
fi

absolute_path=$(readlink -f $config_path)
echo "Using configuration script at $absolute_path"

# Call the Python script to get the 'hf_home' configuration
HF_HOME=$(python $config_path hf_home)

# Check if HF_HOME is empty
if [ -z "$HF_HOME" ]; then
    echo "Failed to load HF_HOME configuration."
    exit 1
fi

echo "Using Hugging Face home directory at $HF_HOME"
export HF_HOME=$HF_HOME

VENV=$(python $config_path venv)
CONDA=$(python $config_path conda)

# Check if VENV is defined
if [ -n "$VENV" ]; then
    # VENV is defined, so we use it
    absolute_path=$(readlink -f "$VENV")
    echo "Using VENV at $absolute_path"
    if [ -f "$absolute_path" ]; then
        source "$absolute_path"
        echo "Virtual environment activated."
    else
        echo "Error: VENV file not found at $absolute_path"
        exit 1
    fi
# If VENV is not defined, check if CONDA is defined
elif [ -n "$CONDA" ]; then
    echo "Using Conda environment"
    # Ensure conda command is available
    if command -v conda >/dev/null 2>&1; then
        # shellcheck disable=SC1091
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate "$CONDA"
        echo "Conda environment '$CONDA' activated."
    else
        echo "Error: conda command not found"
        exit 1
    fi
else
    echo "Error: Neither VENV nor CONDA variable is defined."
    exit 1
fi
