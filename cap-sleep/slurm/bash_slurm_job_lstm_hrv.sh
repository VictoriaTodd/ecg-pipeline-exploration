#!/bin/bash

#SBATCH --job-name=LSTMHRV          # Set the name of your job
#SBATCH --output=slurm_task_%j.log      # Create a log for this job.  This file will be created in the same folder as this script file and include the job number in the filename
#SBATCH --ntasks=1                      # We are planning to run one task
#SBATCH --nodes=1                       # We request the use of 1 compute node for this task
#SBATCH --cpus-per-task=2               # We request the use of 2 CPU's for this single task
#SBATCH --mem-per-cpu=1000mb             # We request a memory allocation of 500mb per CPU
#SBATCH --partition=gpu                 # Request the use of the "gpu" partition | You could also choose the "Virtual" partition for vCPU's only
#SBATCH --gres=gpu:2                    # The number of GPU's we want to use in this partition (Do not include this if choosing "Virtual" partition)
#SBATCH --time=5-0:0:0                 # Time we expect to use these resources for (days - hrs : mins : seconds)

# -----------------------------------------------------------------------------------------------
T1=$(date +%s)                          # Record how long all this takes to complete
# ---------------------------------------------------------------------------------------------

#Let's gather some of the Environment information that we set at the start of this file

echo "HOST environment information"                                         > env_info.txt
echo "Task run at $(date)"                                                  >> env_info.txt
echo "Running on $(hostname -s)"                                            >> env_info.txt
echo "Working directory is $(pwd)"                                          >> env_info.txt
echo "Number of CPU's allocated to this task = $SLURM_CPUS_PER_TASK"        >> env_info.txt
echo "Memory allocated to each CPU = $SLURM_MEM_PER_CPU"                    >> env_info.txt

# We will now run a Python script and use Tensorflow to report on the number of GPU's 
# it can find on this host.

module purge                            # Start with a CLEAN lmod environment
module load cuda/12.6
pip3 install torch torchvision torchaudio
pip3 install mne
source sklearn-env/bin/activate
pip3 install -U scikit-learn
python -W "ignore" test_lstm_hrv.py     # Run our script
wait                                    # Wait until processing has completed
module purge                            # Unload what we've used / clean up after ourselves

#sleep 30

# #########################################################################################
# How long did this take?
T2=$(date +%s)
ELAPSED_TIME=$((T2 - T1))
echo >> env_info.txt
echo "Script has taken $ELAPSED_TIME second(s) to complete" >> env_info.txt
# #########################################################################################
