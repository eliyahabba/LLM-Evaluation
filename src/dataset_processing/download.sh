#!/bin/bash
#SBATCH --gres=gpu:1,vmem:10g
#SBATCH -c24
#SBATCH --mem-per-cpu=10g
#SBATCH --time=1-0

#SBATCH --mail-user=eliya.habba@mail.huji.ac.il
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --exclude=gringolet-01,gringolet-02,gringolet-03,gringolet-04,gringolet-05,gringolet-06

export PYTHONPATH=/cs/labs/gabis/eliyahabba/LLM-Evaluation/
export HF_HOME=/cs/snapless/gabis/gabis/shared/huggingface/
echo "HF_HOME is set to: $HF_HOME"
sacct -j $SLURM_JOB_ID --format=User,JobID,Jobname,partition,state,time,start,end,elapsed,MaxRss,MaxVMSize,nnodes,ncpus,nodelist
dir="/cs/labs/gabis/eliyahabba/LLM-Evaluation/src/dataset_processing/"
absolute_path=$(readlink -f $dir)
# print the full (not relative) path of the dir variable
echo "current dir is set to: $absolute_path"
cd $dir
echo ${SLURM_ARRAY_TASK_ID}
export UNITXT_ALLOW_UNVERIFIED_CODE="True"

python download.py
