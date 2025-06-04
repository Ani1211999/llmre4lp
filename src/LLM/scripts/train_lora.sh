#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

nohup deepspeed fastchat/train/train_lora.py \
    --model_name_or_path lmsys/vicuna-7b-v1.5  \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --data_path /home/cc7738/ws_aniket/LLM_for_Het/LLM4HeG/llm_pred/prompt_json/Amazon/train_all.json \
    --output_dir /home/cc7738/ws_aniket/LLM_for_Het/LLM4HeG/stage1_results_amazon/ \
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
    --flash_attn False  > amazon_train_final.txt 2>&1 &