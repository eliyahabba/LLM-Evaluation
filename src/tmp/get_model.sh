#!/bin/bash

#SBATCH --mem=10g
#SBATCH --time=2:0:0
#SBATCH --gres=gpu:a6000:1
#SBATCH --mail-user=eliya.habba@mail.huji.ac.il
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --exclude=cortex-03,cortex-04,cortex-05,cortex-06,cortex-07,cortex-08
#SBATCH --job-name=calculate_perplexity
#SBATCH --killable


# Now HF_HOME is available to use in this script

dir="/Users/ehabba/PycharmProjects/LLM-Evaluation/src/tmp/"
absolute_path=$(readlink -f $dir)
# print the full (not relative) path of the dir variable
echo "current dir is set to: $absolute_path"
cd $dir

sacct -j $SLURM_JOB_ID --format=User,JobID,Jobname,partition,state,time,start,end,elapsed,MaxRss,MaxVMSize,nnodes,ncpus,nodelist
module load cuda
module load torch
echo ${SLURM_ARRAY_TASK_ID}
export UNITXT_ALLOW_UNVERIFIED_CODE="True"
CUDA_LAUNCH_BLOCKING=1 python get_model.py
