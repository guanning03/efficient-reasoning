CUDA_VISIBLE_DEVICES=0 python evaluate_model.py \
    --model_path='deepseek-ai/DeepSeek-R1-Distill-Qwen-7B' \
    --dataset='Maxwell-Jia/AIME_2024' \
    --scale=7B
