#!/bin/bash
#SBATCH --mem=2g
#SBATCH -c4
#SBATCH --time=0-6
#SBATCH --mail-user=eliya.habba@mail.huji.ac.il
#SBATCH --mail-type=END,FAIL,TIME_LIMIT



load_config_path="load_config.sh"
config_bash=$(readlink -f $load_config_path)
echo "Loading config with: " $config_bash
source $config_bash

# Now HF_HOME is available to use in this script
echo "HF_HOME is set to: $HF_HOME"
export PYTHONPATH=/cs/labs/gabis/eliyahabba/LLM-Evaluation/

sacct -j $SLURM_JOB_ID --format=User,JobID,Jobname,partition,state,time,start,end,elapsed,MaxRss,MaxVMSize,nnodes,ncpus,nodelist
module load cuda
module load torch

echo ${SLURM_ARRAY_TASK_ID}
export UNITXT_ALLOW_UNVERIFIED_CODE="True"
CUDA_LAUNCH_BLOCKING=1 python evaluate_results.py --model_index $1
