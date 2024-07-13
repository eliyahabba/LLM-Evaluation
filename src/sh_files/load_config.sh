#!/bin/bash

# Example of how to use the Python script to get the 'hf_home' configuration
# Assuming the Python script is two levels up from the current directory
config_path="../../config/get_config.py"
echo "Using configuration script at $config_path"
# Ensure the Python script exists and is executable
if [ ! -f "$config_path" ]; then
    echo "Configuration script not found at $config_path"
    exit 1
fi

# Call the Python script to get the 'hf_home' configuration
HF_HOME=$(python $config_path hf_home)

# Check if HF_HOME is empty
if [ -z "$HF_HOME" ]; then
    echo "Failed to load HF_HOME configuration."
    exit 1
fi

echo "Using Hugging Face home directory at $HF_HOME"
export HF_HOME=$HF_HOME


# Check if HF_HOME is empty
if [ -z "$VENV" ]; then
    echo "Failed to load VENV configuration."
    exit 1
fi

echo "Using VENV home directory at $VENV"
export VENV=$VENV
