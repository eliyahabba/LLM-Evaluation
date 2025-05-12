#!/bin/bash

#SBATCH --mem=200g
#SBATCH --time=2:0:0
#SBATCH --mail-user=eliya.habba@mail.huji.ac.il
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --job-name=calculate_perplexity
#SBATCH --killable

export HF_HOME=/cs/snapless/gabis/gabis/shared/huggingface/

# Now HF_HOME is available to use in this script

#dir="/Users/ehabba/PycharmProjects/LLM-Evaluation/src/tmp/"
#absolute_path=$(readlink -f $dir)
# print the full (not relative) path of the dir variable
#echo "current dir is set to: $absolute_path"
#cd $dir

PROJECT_DIR="/cs/labs/gabis/eliyahabba/LLM-Evaluation/src/tmp/"
cd $PROJECT_DIR
echo "Current directory: $(pwd)"

#source "/cs/snapless/gabis/eliyahabba/venvs/LLM-Evaluation/bin/activate"
#echo "Virtual environment activated"

sacct -j $SLURM_JOB_ID --format=User,JobID,Jobname,partition,state,time,start,end,elapsed,MaxRss,MaxVMSize,nnodes,ncpus,nodelist
module load cuda
module load torch
echo ${SLURM_ARRAY_TASK_ID}
export UNITXT_ALLOW_UNVERIFIED_CODE="True"
CUDA_LAUNCH_BLOCKING=1 python get_model.py