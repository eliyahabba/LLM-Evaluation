#!/bin/bash

#SBATCH --job-name=llama3-quantized
#SBATCH --mem=32g
#SBATCH --time=5:0:0
#SBATCH --mail-user=eliya.habba@mail.huji.ac.il
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --killable
#SBATCH --requeue

# Set Hugging Face cache directory
export HF_HOME=/cs/snapless/gabis/gabis/shared/huggingface/

# Set project directory
PROJECT_DIR="/cs/labs/gabis/eliyahabba/LLM-Evaluation/src/tmp/"
cd $PROJECT_DIR
echo "Current directory: $(pwd)"

# Load modules
module load cuda
module load torch

# Print job info
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Task ID (for array jobs): ${SLURM_ARRAY_TASK_ID}"

# Allow running of unverified code
export UNITXT_ALLOW_UNVERIFIED_CODE="True"
export CUDA_LAUNCH_BLOCKING=1

# Run the Python script
echo "Starting model loading and testing..."
python get_model.py

# Print resource usage at the end
echo "Job resource usage:"
sacct -j $SLURM_JOB_ID --format=User,JobID,Jobname,partition,state,time,start,end,elapsed,MaxRss,MaxVMSize,nnodes,ncpus,nodelist