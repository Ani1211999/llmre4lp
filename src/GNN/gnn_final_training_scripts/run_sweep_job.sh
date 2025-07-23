#!/bin/bash
#SBATCH --job-name=gnn_train
#SBATCH --output=training_output_gnn_%j.txt
#SBATCH --error=training_error_gnn_%j.txt
#SBATCH --partition=4090
#SBATCH --gres=gpu:1
#SBATCH --time=35:00:00   # More than 30 hours
# Check if deepspeed is available

source /mnt/webscistorage/cc7738/anaconda3/etc/profile.d/conda.sh
conda activate llm4heg-env
module load cuda/12.1
module load gcc
module list
export CUDA_VISIBLE_DEVICES=0
export PATH=/usr/local/gcc/12/bin:$PATH
mkdir -p $HOME/bin
ln -sf /usr/local/gcc/12/bin/g++ $HOME/bin/c++
export PATH=$HOME/bin:$PATH

which c++
which g++


# Check if deepspeed is installed and show version

python gnn_final_training_scripts/sweep_runner_script.py
