#!/bin/bash

# 创建logs目录
mkdir -p logs

# 获取当前时间戳
timestamp=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/run_${timestamp}.out"
ERR_FILE="logs/run_${timestamp}.err"

export CUDA_VISIBLE_DEVICES="0,1,2,3"
export WANDB_MODE="disabled"

echo "job is starting on `hostname`" | tee -a $LOG_FILE

NUM_GPUS=4

MODEL_SCALE='1.5B'
REWARD_TYPE='sigmoid'
ALPHA=0.1
WANDB_KEY="256879fdda25bc1fb8ee4f0310e71615e92f75c9"
CHECK_EOS='--check_eos'
SCHEDULER_TYPE='warmup_with_constant_lr'

PRETRAIN='DeepSeek-R1-Distill-Qwen-1.5B'  
ACTOR_NUM_GPUS=2
REF_NUM_GPUS=2
VLLM_NUM_ENGINES=2
ACTOR_LEARNING_RATE=5e-6
INIT_KL_COEF=0.001
MIN_P=0
MAX_EPOCHS=1
TOKENIZER='DeepSeek-R1-Distill-Qwen-1.5B'
NUM_EPISODES=1
GENERATE_MAX_LEN=32000
SAVE_STEPS=10
SEED=42

RUN_NAME="scale:"$MODEL_SCALE"_alpha:"$ALPHA
INPUT_KEY="problem"
DATASET="benchmarks/gsm8k"
BASE_PROJECT_DIR="."
RM_ADDRESS="0.0.0.0:24372"
SAVE_PATH="$BASE_PROJECT_DIR/$RUN_NAME"
CKPT_PATH="$SAVE_PATH"

echo "Using: ($DATASET) logging run to ($RUN_NAME)" | tee -a $LOG_FILE

# 停止已有的Ray实例
ray stop
# 启动Ray
ray start --head --node-ip-address 0.0.0.0 --num-gpus $NUM_GPUS --ray-debugger-external

# 启动reward server
python -m reward_server.math_server \
  --address $RM_ADDRESS \
  --dataset $DATASET \
  --tokenizer $TOKENIZER \
  --reward_type $REWARD_TYPE \
  --alpha $ALPHA \
  $CHECK_EOS \
  >> logs/reward_server.out 2>> logs/reward_server.err &

# 主训练程序
python -m openrlhf.cli.train_ppo_ray \
  --advantage_estimator rloo \
  --n_samples_per_prompt 8 \
  --max_epochs $MAX_EPOCHS \
  --remote_rm_url http://$RM_ADDRESS/query \
  --ref_num_nodes 1 \
  --ref_num_gpus_per_node $REF_NUM_GPUS \
  --actor_num_nodes 1 \
  --actor_num_gpus_per_node $ACTOR_NUM_GPUS \
  --vllm_num_engines $VLLM_NUM_ENGINES \
  --vllm_tensor_parallel_size 1 \
  --max_ckpt_num 10 \
  --num_episodes $NUM_EPISODES \
  --colocate_critic_reward \
  --colocate_actor_ref \
  --pretrain $PRETRAIN \
  --wandb_run_name $RUN_NAME \
  --save_path $SAVE_PATH \
  --ckpt_path $CKPT_PATH \
  --save_steps $SAVE_STEPS \
  --prompt_data_probs 1.0 \
  --scheduler_type $SCHEDULER_TYPE \
  --min_p $MIN_P \
  --micro_train_batch_size 1 \
  --train_batch_size 128 \
  --micro_rollout_batch_size 1 \
  --rollout_batch_size 32 \
  --max_samples 3200 \
  --prompt_max_len 512 \
  --generate_max_len $GENERATE_MAX_LEN \
  --zero_stage 2 \
  --bf16 \
  --seed $SEED \
  --actor_learning_rate $ACTOR_LEARNING_RATE \
  --init_kl_coef $INIT_KL_COEF \
  --prompt_data $DATASET \
  --input_key $INPUT_KEY \
  --input_template $'<|im_start|>user\nPlease reason step by step and put your final answer after ####. Question: {}\n<|im_end|>\n<|im_start|>assistant\n' \
  --flash_attn \
  --gradient_checkpointing \
  --use_wandb $WANDB_KEY \
  >> $LOG_FILE 2>> $ERR_FILE

echo "Job completed" | tee -a $LOG_FILE