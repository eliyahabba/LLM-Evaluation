#!/bin/bash

#SBATCH --mem=12g
#SBATCH --time=6:0:0
#SBATCH --gres=gpu:1,vmem:24
#SBATCH --mail-user=eliya.habba@mail.huji.ac.il
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --exclude=cortex-03,cortex-04,cortex-05,cortex-06,cortex-07,cortex-08
#SBATCH --killable

python_path="../../"
export PYTHONPATH=$python_path
sacct -j $SLURM_JOB_ID --format=User,JobID,Jobname,partition,state,time,start,end,elapsed,MaxRss,MaxVMSize,nnodes,ncpus,nodelist
module load cuda
module load torch

dir="../experiments/"
cd $dir

echo "VENV is set to: $VENV"
source $VENV

echo ${SLURM_ARRAY_TASK_ID}
CUDA_LAUNCH_BLOCKING=1 python run_experiment.py --model_name GEMMA2_9B --card $1 --template_range $2 $3     --num_demos $4 --demos_pool_size $5
