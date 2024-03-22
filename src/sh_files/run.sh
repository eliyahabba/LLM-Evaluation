#!/bin/bash
#SBATCH --mem=30g
#SBATCH --time=0-3
#SBATCH --gres=gpu:1,vmem:48g
#SBATCH --mail-user=eliya.habba@mail.huji.ac.il
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --exclude=ampere-01,arion-01,arion-01,arion-02,binky-01,binky-02,binky-03,binky-04,binky-05
export HF_HOME="/cs/snapless/gabis/gabis/shared/huggingface"
export PYTHONPATH=/cs/labs/gabis/eliyahabba/LLM-Evaluation/

sacct -j $SLURM_JOB_ID --format=User,JobID,Jobname,partition,state,time,start,end,elapsed,MaxRss,MaxVMSize,nnodes,ncpus,nodelist
module load cuda
module load torch

dir=/cs/labs/gabis/eliyahabba/LLM-Evaluation/src/Experiments/
cd $dir

source /cs/snapless/gabis/eliyahabba/venvs/LLM-Evaluation/bin/activate

echo ${SLURM_ARRAY_TASK_ID}
CUDA_LAUNCH_BLOCKING=1 python run_experiment.py --template_num $1