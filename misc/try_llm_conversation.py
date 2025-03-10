from vllm import LLM, SamplingParams

model = LLM('./DeepSeek-R1-Distill-Qwen-7B', max_model_len=8192)

# 定义颜色代码
BLUE = '\033[94m'    # 用户文本颜色
GREEN = '\033[92m'   # 助手文本颜色
RESET = '\033[0m'    # 重置颜色

# 设置采样参数
sampling_params = SamplingParams(temperature=0.7, max_tokens=1024)

# 打印带颜色的问题

Question = "What is the capital of France?"
print(f"{BLUE}User: {Question}{RESET}")

# 获取模型回答
response = model.generate([Question], sampling_params)[0].outputs[0].text.strip()

# 打印带颜色的回答
print(f"{GREEN}Assistant: {response}{RESET}")

import pdb; pdb.set_trace()
print('done')