#!/bin/bash

#SBATCH --mem=100g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --job-name=run
#SBATCH --partition=ghx4
#SBATCH --time=24:00:00      # hh:mm:ss for the job
#SBATCH -e logs/slurm-%j.err
#SBATCH -o logs/slurm-%j.out

export CUDA_VISIBLE_DEVICES=0
export WANDB_MODE=disabled
echo "job is starting on `hostname`"

# NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
NUM_GPUS=1

REWARD_TYPE='sigmoid'
ALPHA=0.0 # This controls the penalty for longer correct respones. Increase to penalize longer responses.
WANDB_KEY="256879fdda25bc1fb8ee4f0310e71615e92f75c9" # Provide your wandb key here before running
CHECK_EOS='--check_eos'
SCHEDULER_TYPE='warmup_with_constant_lr' # can be cosine otherwise

ROLLOUT_BATCH_SIZE=4

PRETRAIN='ckpt/sft_qwen_gsm8k'  
ACTOR_NUM_GPUS=1
REF_NUM_GPUS=1
VLLM_NUM_ENGINES=1
ACTOR_LEARNING_RATE=5e-6
INIT_KL_COEF=0.001
MIN_P=0.0
MAX_EPOCHS=2
TOKENIZER='Qwen2.5-0.5B'
NUM_EPISODES=10
GENERATE_MAX_LEN=8192
SAVE_STEPS=10
SEED=42

RUN_NAME="Qwen2.5-0.5B_rollout_$ROLLOUT_BATCH_SIZE"
INPUT_KEY="problem"
DATASET='openai/gsm8k'
BASE_PROJECT_DIR=ckpt/checkpoints_ppo/ # Change this to the path of the project directory
RM_ADDRESS="0.0.0.0:24372"
SAVE_PATH="$BASE_PROJECT_DIR/$RUN_NAME"
CKPT_PATH="$SAVE_PATH"

echo "Using: ($DATASET) logging run to ($RUN_NAME)"

# stop if any previous instances are running
ray stop
# launch the master node of ray in container
ray start --head --node-ip-address 0.0.0.0 --num-gpus $NUM_GPUS --ray-debugger-external

# launch reward server
python -m reward_server.math_server \
  --address $RM_ADDRESS \
  --dataset $DATASET \
  --tokenizer $TOKENIZER \
  --reward_type $REWARD_TYPE \
  --alpha $ALPHA \
  $CHECK_EOS \
  1> logs/server.out 2> logs/server.err&

python -m openrlhf.cli.train_ppo_ray \
  --advantage_estimator gae \
  --n_samples_per_prompt 4 \
  --max_epochs $MAX_EPOCHS \
  --remote_rm_url http://$RM_ADDRESS/query \
  --ref_num_nodes 1 \
  --ref_num_gpus_per_node $REF_NUM_GPUS \
  --actor_num_nodes 1 \
  --actor_num_gpus_per_node $ACTOR_NUM_GPUS \
  --critic_num_nodes 1 \
  --critic_num_gpus_per_node 1 \
  --vllm_num_engines $VLLM_NUM_ENGINES \
  --vllm_tensor_parallel_size 1 \
  --max_ckpt_num 10 \
  --num_episodes $NUM_EPISODES \
  --colocate_all_models \
  --pretrain $PRETRAIN \
  --wandb_run_name $RUN_NAME \
  --save_path $SAVE_PATH \
  --ckpt_path $CKPT_PATH \
  --save_steps $SAVE_STEPS \
  --prompt_data_probs 1.0 \
  --scheduler_type $SCHEDULER_TYPE \
  --min_p $MIN_P \
  --micro_train_batch_size 1 \
  --train_batch_size 16 \
  --micro_rollout_batch_size 1 \
  --rollout_batch_size $ROLLOUT_BATCH_SIZE \
  --max_samples 10000 \
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
  --use_wandb $WANDB_KEY