#!/bin/bash
#SBATCH --mem=2g
#SBATCH -c4
#SBATCH --time=0-6
#SBATCH --mail-user=eliya.habba@mail.huji.ac.il
#SBATCH --mail-type=END,FAIL,TIME_LIMIT



export HF_HOME="/cs/snapless/gabis/gabis/shared/huggingface"
export PYTHONPATH=/cs/labs/gabis/eliyahabba/LLM-Evaluation/

sacct -j $SLURM_JOB_ID --format=User,JobID,Jobname,partition,state,time,start,end,elapsed,MaxRss,MaxVMSize,nnodes,ncpus,nodelist
module load cuda
module load torch

echo "VENV is set to: $VENV"
source $VENV

echo ${SLURM_ARRAY_TASK_ID}
CUDA_LAUNCH_BLOCKING=1 python evaluate_results.py --model_index $1
