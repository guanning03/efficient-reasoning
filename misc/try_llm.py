from vllm import LLM, SamplingParams

model = LLM('./DeepSeek-R1-Distill-Qwen-7B', max_model_len = 8192)

sampling_params = SamplingParams(
    temperature=0.7,
    max_tokens=1024
)

# 提问
prompt = "What is the capital of France?"
outputs = model.generate([prompt], sampling_params)

# 打印回答
for output in outputs:
    print("Question:", prompt)
    print("Answer:", output.outputs[0].text)
    print("\nGenerated tokens:", len(output.outputs[0].token_ids))