import os, time
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import json
from vllm import LLM, SamplingParams
from datasets import load_from_disk, load_dataset
from utils import DATASET_KEYS, RESPONSE_EXTRACTOR, RESPONSE_COMPARATOR
import pandas as pd
import argparse
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

os.makedirs('results', exist_ok=True)
os.makedirs('outputs', exist_ok=True)

# 设置全局变量
base_model_path = 'Qwen2.5-0.5B'
ckpt_base_path = "ckpt/checkpoints_sft/global_step{}/mp_rank_00_model_states.pt"
# ckpt_path = None
ckpt_shortname = 'qwen_gsm8k_sft_step0'
dataset_name = "openai/gsm8k"

parser = argparse.ArgumentParser()
# parser.add_argument('--dataset', type=str)
parser.add_argument('--tok_limit', type=int, default=8192)
args = parser.parse_args()
os.environ['TOKENIZERS_PARALLELISM'] = "false"
split = 'test'

# dataset_name = args.dataset
tok_limit = args.tok_limit
results = {}

print("Dataset:", dataset_name)
print("Base model:", base_model_path)
print("Checkpoint:", ckpt_base_path)

QUESTION_KEY = DATASET_KEYS[dataset_name]["question"]
ANSWER_KEY = DATASET_KEYS[dataset_name]["answer"]
eq = RESPONSE_COMPARATOR[dataset_name]
if not ckpt_base_path:
    ckpt_shortname = base_model_path

if dataset_name == 'datasets/converted_aime_dataset':
    dataset = load_from_disk(dataset_name)
    TEST_N = 10
    MAX_TOKENS = tok_limit
    TEST_TEMPERATURE = 0.6
    MAX_TEST_SAMPLES = 100
elif dataset_name == 'di-zhang-fdu/MATH500':
    dataset = load_dataset(dataset_name)
    TEST_N = 3
    MAX_TOKENS = tok_limit
    TEST_TEMPERATURE = 0.6
    MAX_TEST_SAMPLES = 500
elif dataset_name == 'openai/gsm8k':
    dataset = load_dataset(dataset_name, 'main')
    TEST_N = 1
    MAX_TOKENS = tok_limit
    TEST_TEMPERATURE = 0.6
    MAX_TEST_SAMPLES = 1319
if dataset_name == 'Maxwell-Jia/AIME_2024':
    dataset = load_from_disk('benchmarks/AIME2024')
    print("\nDataset columns:", dataset['train'].column_names)  # 添加这行
    TEST_N = 5
    MAX_TOKENS = tok_limit
    TEST_TEMPERATURE = 0.6
    MAX_TEST_SAMPLES = 30
elif dataset_name == 'opencompass/AIME2025':
    dataset = load_from_disk('benchmarks/AIME2025')
    print("\nDataset columns:", dataset['train'].column_names)  # 添加这行
    TEST_N = 5
    MAX_TOKENS = tok_limit
    TEST_TEMPERATURE = 0.6
    MAX_TEST_SAMPLES = 15

def get_scores(ds, outputs, save_file_name=None):
    predictions, golds = [], []
    results = []
    for input, output in zip(ds, outputs):
        gold = RESPONSE_EXTRACTOR[dataset_name](input[ANSWER_KEY])
        prediction = [
            RESPONSE_EXTRACTOR[dataset_name](resp.text)
            for resp in output.outputs
        ]
        predictions.append(prediction)
        golds.append(gold)
        results.append(
            {
                QUESTION_KEY: input[QUESTION_KEY],
                ANSWER_KEY: input[ANSWER_KEY],
                "responses": [resp.text for resp in output.outputs],
                "prediction": prediction,
                "gold": gold,
                "tokens": sum([len(resp.token_ids) for resp in output.outputs]) / len(output.outputs),
                "accuracy": [eq(gold, pred) for pred in prediction],
            }
        )
    if save_file_name is not None:
        print(f"正在保存结果到: {save_file_name}")
        with open(save_file_name, 'w') as f:
            json.dump(results, f, indent=4)

    results = pd.DataFrame(results)
    predictions, golds, tokens = results["prediction"], results["gold"], results["tokens"]
    pass_at_1 = sum([any([eq(g, pred) for pred in p[:1]]) for p, g in zip(predictions, golds)]) / len(predictions)
    pass_at_k_list = []
    acc_at_k_list = []
    k = TEST_N
    print("Average tokens:", sum(tokens) / len(tokens))
    for i in range(k):
        pass_at_i = sum([any([eq(g, pred) for pred in p[:i+1]]) for p, g in zip(predictions, golds)]) / len(predictions)
        acc_at_i = sum([eq(g, p[i]) for p, g in zip(predictions, golds)]) / len(predictions)
        acc_at_k_list.append(acc_at_i)
        pass_at_k_list.append(pass_at_i)
        print(
            f"Pass @ {i+1}: {pass_at_i}"
        )

    def get_most_common(solns):
        soln_counts = {}
        for soln in solns:
            if soln is None:
                continue
            added = False
            for other_solns in solns:
                if eq(soln, other_solns):
                    added = True
                    soln_counts[soln] = soln_counts.get(soln, 0) + 1
            if not added:
                soln_counts[soln] = 1
        if len(soln_counts) == 0:
            return None
        return max(soln_counts, key=soln_counts.get)
    
    predictions_maj = [get_most_common(p) for p in predictions]
    all_preds = sum([[eq(golds[i], p) for p in predictions[i]] for i in range(len(predictions))], [])
    avg_pass_rate = sum(all_preds) / len(all_preds)
    pass_at_n = sum([eq(g, p) for p, g in zip(predictions_maj, golds)]) / len(predictions)
    print(
        f"Pass @ 1(with majority): {pass_at_n}"
    )
    
    return {
        'pass@1': pass_at_1,
        'pass@1(majority)': sum([eq(g, p) for p, g in zip(predictions_maj, golds)]) / len(predictions),
        'average_pass_rate': avg_pass_rate,
        'std_pass_rate': np.std(acc_at_k_list),
        'acc@k': acc_at_k_list,
        'pass@k': pass_at_k_list,
        'avg_tokens': sum(tokens) / len(tokens)
    }


def evaluate_model(ckpt_path, step, split):
    test_prompts = []
    
    print(f"正在评估 split: {split}")
    
    # 首先用transformers加载基础模型和tokenizer
    print("正在加载基础模型...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        device_map="cpu",
        trust_remote_code=True
    )
    
    if ckpt_path:
        print(f"正在加载checkpoint: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        if "module" in checkpoint:
            checkpoint = checkpoint["module"]
        base_model.load_state_dict(checkpoint, strict=False)
    
    # 将更新后的权重保存为临时文件
    temp_model_path = "temp_model"
    print("正在保存更新后的模型...")
    base_model.save_pretrained(temp_model_path)
    del base_model  # 释放内存
    torch.cuda.empty_cache()
    
    # 使用vLLM加载更新后的模型
    print("正在初始化vLLM...")
    model = LLM(
        model=temp_model_path,  # 使用保存的临时模型
        tokenizer=base_model_path,  # 仍使用原始tokenizer
        gpu_memory_utilization=0.7,  # 将显存使用率设置为60%
        tensor_parallel_size=1,
        max_model_len=MAX_TOKENS + 8192,
        swap_space=80
    )

    # 使用指定的split
    test_ds = dataset[split].shuffle(seed=0).select(range(min(MAX_TEST_SAMPLES, len(dataset[split]))))

    for x in test_ds:
        prompt = f"<|im_start|>user\nPlease reason step by step and put your final answer after ####. Question: {x[QUESTION_KEY]}\n<|im_end|>\n<|im_start|>assistant\n"
        prompt_tokens = model.llm_engine.tokenizer.tokenizer.encode(prompt)
        test_prompts.append(prompt_tokens)
    
    sampling_params = SamplingParams(
        temperature=TEST_TEMPERATURE,
        max_tokens=MAX_TOKENS,
        n=TEST_N
    )
    sampling_params.stop_token_ids = [model.llm_engine.tokenizer.tokenizer.eos_token_id]
    print("Generating test outputs...")
    print(model.llm_engine.tokenizer.tokenizer.decode(test_prompts[0], skip_special_tokens=False))
    start_time = time.time()
    # 根据step更新ckpt_shortname
    global ckpt_shortname
    ckpt_shortname = f'qwen_gsm8k_sft_step{step}'
    
    test_outputs = model.generate(prompt_token_ids=test_prompts, sampling_params=sampling_params, use_tqdm=True)
    end_time = time.time()
    output_file = f"outputs/{dataset_name.replace('/', '_')}-{split}_results_{ckpt_shortname}_{tok_limit}.json"
    print(f"正在保存输出结果到: {output_file}")
    test_scores = get_scores(test_ds, test_outputs, output_file)
    print("Test:", test_scores)
    time_taken = end_time - start_time
    print("Time taken:", time_taken)

    return {'scores': test_scores, 'time_taken': time_taken}

print("开始评估模型...")
all_results = {}

# 首先评估基础模型（step 0）
print(f"\n评估 step 0 (基础模型)")
ckpt_path = None
all_results[0] = {
    'train': evaluate_model(ckpt_path, 0, 'train'),
    'test': evaluate_model(ckpt_path, 0, 'test')
}

# 评估所有checkpoint
for step in range(276, 1381, 276):
    print(f"\n评估 step {step}")
    ckpt_path = ckpt_base_path.format(step)
    if os.path.exists(ckpt_path):
        all_results[step] = {
            'train': evaluate_model(ckpt_path, step, 'train'),
            'test': evaluate_model(ckpt_path, step, 'test')
        }
    else:
        print(f"警告：找不到checkpoint {ckpt_path}")

# 保存所有结果
save_path = 'results/qwen_sft_gsm8k_batch.json'
print(f"正在保存总结果到: {save_path}")
with open(save_path, 'w') as f:
    json.dump(all_results, f, indent=4)

print(f"\n所有结果已保存到 {save_path}")
