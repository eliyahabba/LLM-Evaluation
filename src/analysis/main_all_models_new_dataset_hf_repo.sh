#!/bin/bash
#SBATCH -c50
#SBATCH --mem-per-cpu=1g
#SBATCH --time=1-0
#SBATCH --gres=gpu:1,vmem:10g

#SBATCH --mail-user=eliya.habba@mail.huji.ac.il
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --exclude=gringolet-01,gringolet-02,gringolet-03,gringolet-04,gringolet-05,gringolet-06

export PYTHONPATH=/cs/labs/gabis/eliyahabba/LLM-Evaluation/
export HF_HOME=/cs/snapless/gabis/gabis/shared/huggingface/
echo "HF_HOME is set to: $HF_HOME"

sacct -j $SLURM_JOB_ID --format=User,JobID,Jobname,partition,state,time,start,end,elapsed,MaxRss,MaxVMSize,nnodes,ncpus,nodelist

dir="/cs/labs/gabis/eliyahabba/LLM-Evaluation/src/analysis/"
absolute_path=$(readlink -f $dir)
echo "current dir is set to: $absolute_path"
cd $dir
# Run the dataset merger with lean processing only
python main_all_models_new_dataset_hf_repo.py