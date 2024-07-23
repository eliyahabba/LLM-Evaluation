#!/bin/bash

# Function to submit jobs using sbatch
function submit_job {
    local card=$1
    local start=$2
    local end=$3

    # Create a temporary script for sbatch
    local job_script="sbatch_job_$RANDOM.sh"
    cat <<EOF > $job_script
#!/bin/bash
#SBATCH --mem=12g
#SBATCH --time=0:10:0
#SBATCH --mail-user=eliya.habba@mail.huji.ac.il
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --exclude=cortex-03,cortex-04,cortex-05,cortex-06,cortex-07,cortex-08
#SBATCH --job-name=run_job_$start_$end
#SBATCH --output=logs/slurm_output_%j.log
#SBATCH --killable
#SBATCH --requeue

# Load configuration
load_config_path="load_config.sh"
config_bash=\$(readlink -f \$load_config_path)
echo "Loading config with: " \$config_bash
source \$config_bash

# Set Python path
python_path="../../../"
export PYTHONPATH=\$python_path

module load cuda
module load torch
# Change to the experiments directory
dir="../../experiments/"
absolute_path=\$(readlink -f \$dir)
# Print the full (not relative) path of the dir variable
echo "Current dir is set to: \$absolute_path"
cd \$dir

# Execute the job
CUDA_LAUNCH_BLOCKING=1 python run_experiment.py --model_name LLAMA70B --card $card --template_range $start $end --load_in_8bit
EOF

    # Submit the job
    sbatch $job_script

    # Optional: Remove the script after submission to keep directory clean
    rm $job_script
}

# Main function to read parameters and submit jobs
function main {
    local params_file="params.txt"

    # Read parameters from file and submit jobs
    while IFS= read -r params; do
        echo "Submitting job with parameters: $params"
        # Extract individual parameters
        read -r card start end <<< "$params"
        submit_job $card $start $end
    done < $params_file
}

# Call main function to execute
main
