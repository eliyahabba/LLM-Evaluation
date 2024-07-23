#!/bin/bash

#SBATCH --mem=10g
#SBATCH --time=6:0:0
#SBATCH --gres=gpu:1,vmem:16g
#SBATCH --mail-user=eliya.habba@mail.huji.ac.il
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --exclude=cortex-03,cortex-04,cortex-05,cortex-06,cortex-07,cortex-08


load_config_path="load_config.sh"
config_bash=$(readlink -f $load_config_path)
echo "Loading config with: " $config_bash
source $config_bash

# Now HF_HOME is available to use in this script
echo "HF_HOME is set to: $HF_HOME"
python_path="../../"
export PYTHONPATH=$python_path
sacct -j $SLURM_JOB_ID --format=User,JobID,Jobname,partition,state,time,start,end,elapsed,MaxRss,MaxVMSize,nnodes,ncpus,nodelist
module load cuda
module load torch

dir="../experiments/"
absolute_path=$(readlink -f $dir)
# print the full (not relative) path of the dir variable
echo "current dir is set to: $absolute_path"
cd $dir

CUDA_LAUNCH_BLOCKING=1 python run_experiment.py --model_name GEMMA_7B --card $1 --template_range $2 $3     --num_demos $4 --demos_pool_size $5
