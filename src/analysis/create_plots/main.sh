#!/bin/bash
#SBATCH --mem=100g
#SBATCH -c24
#SBATCH --time=4:0:0

#SBATCH --mail-user=eliya.habba@mail.huji.ac.il
#SBATCH --mail-type=END,FAIL,TIME_LIMIT

# Set HF_HOME environment variable


load_config_path="load_config.sh"
config_bash=$(readlink -f $load_config_path)
echo "Loading config with: " $config_bash
source $config_bash

export HF_HOME=/cs/snapless/gabis/gabis/shared/huggingface/
echo "HF_HOME is set to: $HF_HOME"

export PYTHONPATH=/cs/labs/gabis/eliyahabba/LLM-Evaluation/

sacct -j $SLURM_JOB_ID --format=User,JobID,Jobname,partition,state,time,start,end,elapsed,MaxRss,MaxVMSize,nnodes,ncpus,nodelist
dir="/cs/labs/gabis/eliyahabba/LLM-Evaluation/src/analysis/create_plots"
absolute_path=$(readlink -f $dir)
# print the full (not relative) path of the dir variable
echo "current dir is set to: $absolute_path"
cd $dir

echo ${SLURM_ARRAY_TASK_ID}
export UNITXT_ALLOW_UNVERIFIED_CODE="True"
CUDA_LAUNCH_BLOCKING=1 python main.py