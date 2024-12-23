#!/bin/bash

#SBATCH --mem=12g
#SBATCH --time=1:0:0
#SBATCH --gres=gpu:1,vmem:10g
#SBATCH --mail-user=eliya.habba@mail.huji.ac.il
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --exclude=cortex-03,cortex-04,cortex-05,cortex-06,cortex-07,cortex-08
#SBATCH --job-name=calculate_perplexity
#SBATCH --killable
#SBATCH --requeue




load_config_path="load_config.sh"
config_bash=$(readlink -f $load_config_path)
echo "Loading config with: " $config_bash
source $config_bash

# Now HF_HOME is available to use in this script
python_path="/cs/labs/gabis/eliyahabba/LLM-Evaluation/"
python_path=$(readlink -f $python_path)
export PYTHONPATH=$python_path
echo "PYTHONPATH: " $python_path

dir="/cs/labs/gabis/eliyahabba/LLM-Evaluation/src/experiments/models_predictors"
absolute_path=$(readlink -f $dir)
# print the full (not relative) path of the dir variable
echo "current dir is set to: $absolute_path"
cd $dir

sacct -j $SLURM_JOB_ID --format=User,JobID,Jobname,partition,state,time,start,end,elapsed,MaxRss,MaxVMSize,nnodes,ncpus,nodelist
module load cuda
module load torch
echo ${SLURM_ARRAY_TASK_ID}
export UNITXT_ALLOW_UNVERIFIED_CODE="True"
CUDA_LAUNCH_BLOCKING=1 python PerplexityCalculator.py
