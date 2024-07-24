#!/bin/bash

#SBATCH --mem=12g
#SBATCH --time=6:0:0
#SBATCH --gres=gpu:1,vmem:48g
#SBATCH --mail-user=eliya.habba@mail.huji.ac.il
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --exclude=cortex-03,cortex-04,cortex-05,cortex-06,cortex-07,cortex-08
#SBATCH --job-name=mmlu_job_array
#SBATCH --array=0-133%50   # Adjust the %2 to set maximum concurrent jobs
#SBATCH --output=mmlu_output_%A_%a.log
#SBATCH --killable
#SBATCH --requeue

load_config_path="load_config.sh"
config_bash=$(readlink -f $load_config_path)
echo "Loading config with: " $config_bash
source $config_bash
# Define a function to map array job indices to script parameters
function set_parameters {
    case $1 in
        0) ARGS=("cards.mmlu.abstract_algebra" 0 28);;
        1) ARGS=("cards.mmlu.abstract_algebra" 28 56);;
        2) ARGS=("cards.mmlu.anatomy" 0 28);;
        3) ARGS=("cards.mmlu.anatomy" 28 56);;
        4) ARGS=("cards.mmlu.astronomy" 0 28);;
        5) ARGS=("cards.mmlu.astronomy" 28 56);;
        6) ARGS=("cards.mmlu.business_ethics" 0 28);;
        7) ARGS=("cards.mmlu.business_ethics" 28 56);;
        8) ARGS=("cards.mmlu.clinical_knowledge" 0 28);;
        9) ARGS=("cards.mmlu.clinical_knowledge" 28 56);;
        10) ARGS=("cards.mmlu.college_biology" 0 28);;
        11) ARGS=("cards.mmlu.college_biology" 28 56);;
        12) ARGS=("cards.mmlu.college_chemistry" 0 28);;
        13) ARGS=("cards.mmlu.college_chemistry" 28 56);;
        14) ARGS=("cards.mmlu.college_computer_science" 0 28);;
        15) ARGS=("cards.mmlu.college_computer_science" 28 56);;
        16) ARGS=("cards.mmlu.college_mathematics" 0 28);;
        17) ARGS=("cards.mmlu.college_mathematics" 28 56);;
        18) ARGS=("cards.mmlu.college_medicine" 0 28);;
        19) ARGS=("cards.mmlu.college_medicine" 28 56);;
        20) ARGS=("cards.mmlu.college_physics" 0 28);;
        21) ARGS=("cards.mmlu.college_physics" 28 56);;
        22) ARGS=("cards.mmlu.computer_security" 0 28);;
        23) ARGS=("cards.mmlu.computer_security" 28 56);;
        24) ARGS=("cards.mmlu.conceptual_physics" 0 28);;
        25) ARGS=("cards.mmlu.conceptual_physics" 28 56);;
        26) ARGS=("cards.mmlu.econometrics" 0 28);;
        27) ARGS=("cards.mmlu.econometrics" 28 56);;
        28) ARGS=("cards.mmlu.electrical_engineering" 0 28);;
        29) ARGS=("cards.mmlu.electrical_engineering" 28 56);;
        30) ARGS=("cards.mmlu.elementary_mathematics" 0 28);;
        31) ARGS=("cards.mmlu.elementary_mathematics" 28 56);;
        32) ARGS=("cards.mmlu.formal_logic" 0 28);;
        33) ARGS=("cards.mmlu.formal_logic" 28 56);;
        34) ARGS=("cards.mmlu.global_facts" 0 28);;
        35) ARGS=("cards.mmlu.global_facts" 28 56);;
        36) ARGS=("cards.mmlu.high_school_biology" 0 28);;
        37) ARGS=("cards.mmlu.high_school_biology" 28 56);;
        38) ARGS=("cards.mmlu.high_school_chemistry" 0 28);;
        39) ARGS=("cards.mmlu.high_school_chemistry" 28 56);;
        40) ARGS=("cards.mmlu.high_school_computer_science" 0 28);;
        41) ARGS=("cards.mmlu.high_school_computer_science" 28 56);;
        42) ARGS=("cards.mmlu.high_school_european_history" 0 28);;
        43) ARGS=("cards.mmlu.high_school_european_history" 28 56);;
        44) ARGS=("cards.mmlu.high_school_geography" 0 28);;
        45) ARGS=("cards.mmlu.high_school_geography" 28 56);;
        46) ARGS=("cards.mmlu.high_school_government_and_politics" 0 28);;
        47) ARGS=("cards.mmlu.high_school_government_and_politics" 28 56);;
        48) ARGS=("cards.mmlu.high_school_macroeconomics" 0 28);;
        49) ARGS=("cards.mmlu.high_school_macroeconomics" 28 56);;
        50) ARGS=("cards.mmlu.high_school_mathematics" 0 28);;
        51) ARGS=("cards.mmlu.high_school_mathematics" 28 56);;
        52) ARGS=("cards.mmlu.high_school_microeconomics" 0 28);;
        53) ARGS=("cards.mmlu.high_school_microeconomics" 28 56);;
        54) ARGS=("cards.mmlu.high_school_physics" 0 28);;
        55) ARGS=("cards.mmlu.high_school_physics" 28 56);;
        56) ARGS=("cards.mmlu.high_school_psychology" 0 28);;
        57) ARGS=("cards.mmlu.high_school_psychology" 28 56);;
        58) ARGS=("cards.mmlu.high_school_statistics" 0 28);;
        59) ARGS=("cards.mmlu.high_school_statistics" 28 56);;
        60) ARGS=("cards.mmlu.high_school_us_history" 0 28);;
        61) ARGS=("cards.mmlu.high_school_us_history" 28 56);;
        62) ARGS=("cards.mmlu.high_school_world_history" 0 28);;
        63) ARGS=("cards.mmlu.high_school_world_history" 28 56);;
        64) ARGS=("cards.mmlu.human_aging" 0 28);;
        65) ARGS=("cards.mmlu.human_aging" 28 56);;
        66) ARGS=("cards.mmlu.human_sexuality" 0 28);;
        67) ARGS=("cards.mmlu.human_sexuality" 28 56);;
        68) ARGS=("cards.mmlu.international_law" 0 28);;
        69) ARGS=("cards.mmlu.international_law" 28 56);;
        70) ARGS=("cards.mmlu.jurisprudence" 0 28);;
        71) ARGS=("cards.mmlu.jurisprudence" 28 56);;
        72) ARGS=("cards.mmlu.logical_fallacies" 0 28);;
        73) ARGS=("cards.mmlu.logical_fallacies" 28 56);;
        74) ARGS=("cards.mmlu.machine_learning" 0 28);;
        75) ARGS=("cards.mmlu.machine_learning" 28 56);;
        76) ARGS=("cards.mmlu.management" 0 28);;
        77) ARGS=("cards.mmlu.management" 28 56);;
        78) ARGS=("cards.mmlu.marketing" 0 28);;
        79) ARGS=("cards.mmlu.marketing" 28 56);;
        80) ARGS=("cards.mmlu.medical_genetics" 0 28);;
        81) ARGS=("cards.mmlu.medical_genetics" 28 56);;
        82) ARGS=("cards.mmlu.miscellaneous" 0 28);;
        83) ARGS=("cards.mmlu.miscellaneous" 28 56);;
        84) ARGS=("cards.mmlu.moral_disputes" 0 28);;
        85) ARGS=("cards.mmlu.moral_disputes" 28 56);;
        86) ARGS=("cards.mmlu.moral_scenarios" 0 28);;
        87) ARGS=("cards.mmlu.moral_scenarios" 28 56);;
        88) ARGS=("cards.mmlu.nutrition" 0 28);;
        89) ARGS=("cards.mmlu.nutrition" 28 56);;
        90) ARGS=("cards.mmlu.philosophy" 0 28);;
        91) ARGS=("cards.mmlu.philosophy" 28 56);;
        92) ARGS=("cards.mmlu.prehistory" 0 28);;
        93) ARGS=("cards.mmlu.prehistory" 28 56);;
        94) ARGS=("cards.mmlu.professional_accounting" 0 28);;
        95) ARGS=("cards.mmlu.professional_accounting" 28 56);;
        96) ARGS=("cards.mmlu.professional_law" 0 28);;
        97) ARGS=("cards.mmlu.professional_law" 28 56);;
        98) ARGS=("cards.mmlu.professional_medicine" 0 28);;
        99) ARGS=("cards.mmlu.professional_medicine" 28 56);;
        100) ARGS=("cards.mmlu.professional_psychology" 0 28);;
        101) ARGS=("cards.mmlu.professional_psychology" 28 56);;
        102) ARGS=("cards.mmlu.public_relations" 0 28);;
        103) ARGS=("cards.mmlu.public_relations" 28 56);;
        104) ARGS=("cards.mmlu.security_studies" 0 28);;
        105) ARGS=("cards.mmlu.security_studies" 28 56);;
        106) ARGS=("cards.mmlu.sociology" 0 28);;
        107) ARGS=("cards.mmlu.sociology" 28 56);;
        108) ARGS=("cards.mmlu.us_foreign_policy" 0 28);;
        109) ARGS=("cards.mmlu.us_foreign_policy" 28 56);;
        110) ARGS=("cards.mmlu.virology" 0 28);;
        111) ARGS=("cards.mmlu.virology" 28 56);;
        112) ARGS=("cards.mmlu.world_religions" 0 28);;
        113) ARGS=("cards.mmlu.world_religions" 28 56);;
    esac
    echo $ARGS
}

# Get parameters for the current array job
set_parameters $SLURM_ARRAY_TASK_ID

load_config_path="load_config.sh"
config_bash=$(readlink -f $load_config_path)
echo "Loading config with: " $config_bash
source $config_bash

# Now HF_HOME is available to use in this script
echo "HF_HOME is set to: $HF_HOME"
python_path="../../../"
absolute_python_path=$(readlink -f $python_path)
export PYTHONPATH=$absolute_python_path
# print the full (not relative) path of the dir variable
echo "PYTHONPATH is set to: $PYTHONPATH"

sacct -j $SLURM_JOB_ID --format=User,JobID,Jobname,partition,state,time,start,end,elapsed,MaxRss,MaxVMSize,nnodes,ncpus,nodelist
module load cuda
module load torch
dir="../../experiments/"
absolute_path=$(readlink -f $dir)
# print the full (not relative) path of the dir variable
echo "current dir is set to: $absolute_path"
cd $dir
CUDA_LAUNCH_BLOCKING=1 python run_experiment.py --model_name LLAMA70B --card ${ARGS[0]} --template_range ${ARGS[1]} ${ARGS[2]} --load_in_8bit
