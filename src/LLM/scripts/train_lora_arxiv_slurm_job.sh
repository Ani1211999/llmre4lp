#!/bin/bash
#SBATCH --job-name=lora_train
#SBATCH --output=training_output_%j.txt
#SBATCH --error=training_error_%j.txt
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


nohup deepspeed fastchat/train/train_lora_arxiv.py \
    --model_name_or_path lmsys/vicuna-7b-v1.5  \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --data_path ../../llm_pred/prompt_json/Arxiv/train_all.json \
    --output_dir ../../stage1_results_arxiv_lp_V1/ \
    --num_train_epochs 2 \
    --fp16 True \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --eval_steps 10000  \
    --save_strategy "steps" \
    --save_steps 20000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_strategy "steps" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 4096 \
    --q_lora False \
    --deepspeed ./playground/deepspeed_config_s3.json \
    --gradient_checkpointing True \
    --flash_attn False