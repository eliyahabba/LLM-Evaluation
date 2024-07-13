#!/bin/bash


#SBATCH --mem=10g
#SBATCH --time=2-0
#SBATCH --gres=gpu:1,vmem:45g
#SBATCH --mail-user=eliya.habba@mail.huji.ac.il
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --exclude=cortex-03,cortex-04,cortex-05,cortex-06,cortex-07,cortex-08,drape-01,drape-02,drape-03
#SBATCH --killable


load_config_path="../load_config.sh"
source $load_config_path

# Now HF_HOME is available to use in this script
echo "HF_HOME is set to: $HF_HOME"
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
CUDA_LAUNCH_BLOCKING=1 python run_experiment.py --model_name PHI_MEDIUM --card $1 --template_range $2 $3 --trust_remote_code
