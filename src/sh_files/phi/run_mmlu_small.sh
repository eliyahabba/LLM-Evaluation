#!/bin/bash
#SBATCH --mem=20g
#SBATCH --time=2-0
#SBATCH --gres=gpu:1,vmem:10g
#SBATCH --mail-user=eliya.habba@mail.huji.ac.il
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --exclude=cortex-01,cortex-02,cortex-03,cortex-04,cortex-05,cortex-06,cortex-07,cortex-08
#SBATCH --killable

export HF_HOME="/cs/snapless/gabis/gabis/shared/huggingface"
export PYTHONPATH=/cs/labs/gabis/eliyahabba/LLM-Evaluation/

sacct -j $SLURM_JOB_ID --format=User,JobID,Jobname,partition,state,time,start,end,elapsed,MaxRss,MaxVMSize,nnodes,ncpus,nodelist
module load cuda
module load torch

dir=/cs/labs/gabis/eliyahabba/LLM-Evaluation/src/Experiments/
cd $dir

source /cs/snapless/gabis/eliyahabba/venvs/LLM-Evaluation/bin/activate

echo ${SLURM_ARRAY_TASK_ID}
CUDA_LAUNCH_BLOCKING=1 python run_experiment.py --model_name PHI_SMALL --card $1 --template_range $2 $3 --trust_remote_code --not_load_in_8bit