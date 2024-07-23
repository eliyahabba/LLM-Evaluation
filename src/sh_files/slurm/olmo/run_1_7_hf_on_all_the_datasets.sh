#!/bin/bash

#SBATCH --mem=12g
#SBATCH --time=0:10:0
#SBATCH --gres=gpu:1,vmem:12g
#SBATCH --mail-user=eliya.habba@mail.huji.ac.il
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --exclude=cortex-03,cortex-04,cortex-05,cortex-06,cortex-07,cortex-08
#SBATCH --job-name=mmlu_job_array
#SBATCH --array=0-246%50   # Full data is 246 configurations
#SBATCH --output=logs/slurm_output_%A_%a.log
#SBATCH --killable
#SBATCH --requeue

load_config_path="load_config.sh"
config_bash=$(readlink -f $load_config_path)
echo "Loading config with: " $config_bash
source $config_bash
# Generate parameters based on dataset name, total items, and split size
function generate_params {
    local dataset=$1
    local total=$2
    local split=$3

    for ((i=0; i<total; i+=split)); do
        local end=$((i + split))
        if [ $end -gt $total ]; then
            end=$total
        fi
        ARGS+=("$dataset $i $end")
    done
}

# Define a function to map array job indices to script parameters
function set_parameters {
    local dataset total split
    ARGS=() # Initialize empty array

    # Define custom splits for specified datasets
    declare -A custom_config=(
        ["mmlu.professional_law"]="56 3"
        ["mmlu.high_school_psychology"]="56 10"
        ["mmlu.professional_psychology"]="56 10"
        ["mmlu.miscellaneous"]="56 6"
        ["ai2_arc.arc_challenge"]="56 3"
        ["boolq"]="56 2"
        ["hellaswag"]="56 1"
       )

    # Apply custom configurations
    for key in "${!custom_config[@]}"; do
        IFS=' ' read -r -a config <<< "${custom_config[$key]}"
        generate_params "cards.$key" "${config[0]}" "${config[1]}"
    done

    # Default datasets not specified in custom_config
# Default datasets not specified in custom_config
    # Default datasets not specified in custom_config
    declare -a default_datasets=("mmlu.abstract_algebra" "mmlu.anatomy" "mmlu.astronomy" "mmlu.business_ethics" "mmlu.clinical_knowledge" "mmlu.college_biology" "mmlu.college_chemistry" "mmlu.college_computer_science" "mmlu.college_mathematics" "mmlu.college_medicine" "mmlu.college_physics" "mmlu.computer_security" "mmlu.conceptual_physics" "mmlu.econometrics" "mmlu.electrical_engineering" "mmlu.elementary_mathematics" "mmlu.formal_logic" "mmlu.global_facts" "mmlu.high_school_biology" "mmlu.high_school_chemistry" "mmlu.high_school_computer_science" "mmlu.high_school_european_history" "mmlu.high_school_geography" "mmlu.high_school_government_and_politics" "mmlu.high_school_macroeconomics" "mmlu.high_school_mathematics" "mmlu.high_school_microeconomics" "mmlu.high_school_physics" "mmlu.high_school_us_history" "mmlu.high_school_world_history" "mmlu.human_aging" "mmlu.human_sexuality" "mmlu.international_law" "mmlu.jurisprudence" "mmlu.logical_fallacies" "mmlu.machine_learning" "mmlu.management" "mmlu.marketing" "mmlu.medical_genetics" "mmlu.professional_accounting" "mmlu.professional_medicine" "mmlu.public_relations" "mmlu.security_studies" "mmlu.sociology" "mmlu.us_foreign_policy" "mmlu.virology" "mmlu.world_religions")
    local default_split=28

    for dataset in "${default_datasets[@]}"; do
        if [[ -z "${custom_config[$dataset]}" ]]; then
            generate_params "cards.$dataset" 56 $default_split # Assuming each has 56 items
        fi
    done

    echo "${ARGS[$1]}"
}

# Get parameters for the current array job
PARAMS=$(set_parameters $SLURM_ARRAY_TASK_ID)

python_path="../../../"
export PYTHONPATH=$python_path
sacct -j $SLURM_JOB_ID --format=User,JobID,Jobname,partition,state,time,start,end,elapsed,MaxRss,MaxVMSize,nnodes,ncpus,nodelist
module load cuda
module load torch

dir="../../experiments/"
absolute_path=$(readlink -f $dir)
# print the full (not relative) path of the dir variable
echo "current dir is set to: $absolute_path"
cd $dir

echo ${SLURM_ARRAY_TASK_ID}
read -r card start end <<< "${PARAMS}"
echo ${card}
echo ${start}
echo ${end}
CUDA_LAUNCH_BLOCKING=1 python run_experiment.py --model_name OLMO_1_7 --card $card --template_range $start $end --load_in_8bit
