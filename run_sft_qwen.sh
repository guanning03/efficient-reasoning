#!/bin/bash

#SBATCH --mem=40g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --job-name=sft_qwen
#SBATCH --time=24:00:00
#SBATCH -e logs/slurm-%j.err
#SBATCH -o logs/slurm-%j.out

echo "job is starting on `hostname`"

export MASTER_PORT=$((12000 + RANDOM % 20000))  # 随机生成一个端口号
export CUDA_VISIBLE_DEVICES=3
# export WANDB_MODE=offline
export OMPI_MCA_opal_cuda_support=true

MODEL_PATH="Qwen2.5-0.5B"
DATASET="openai/gsm8k"
WANDB_KEY="256879fdda25bc1fb8ee4f0310e71615e92f75c9" 

# 训练相关参数
LEARNING_RATE=5e-6
MAX_EPOCHS=5
MICRO_BATCH_SIZE=27
TRAIN_BATCH_SIZE=27
MAX_LENGTH=32768
SAVE_STEPS=69
LOGGING_STEPS=1
EVAL_STEPS=69

RUN_NAME="sft_qwen_gsm8k"
SAVE_PATH="./ckpt/$RUN_NAME"

python -m openrlhf.cli.train_sft \
    --pretrain $MODEL_PATH \
    --dataset $DATASET \
    --input_key "question" \
    --output_key "answer" \
    --save_path $SAVE_PATH \
    --max_epochs $MAX_EPOCHS \
    --micro_train_batch_size $MICRO_BATCH_SIZE \
    --train_batch_size $TRAIN_BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --max_len $MAX_LENGTH \
    --save_steps $SAVE_STEPS \
    --logging_steps $LOGGING_STEPS \
    --eval_steps $EVAL_STEPS \
    --zero_stage 2 \
    --bf16 \
    --flash_attn \
    --gradient_checkpointing \
    --input_template $'<|im_start|>user\nPlease reason step by step and put your final answer after ####. Question: {}\n<|im_end|>\n<|im_start|>assistant\n' \
    --wandb_run_name $RUN_NAME \
    --use_wandb $WANDB_KEY \
    --max_ckpt_num 1000 \
    --seed 42
