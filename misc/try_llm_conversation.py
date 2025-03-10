from vllm import LLM, SamplingParams

model = LLM('./Qwen2.5-0.5B', max_model_len = 8192)

sampling_params = SamplingParams(
    temperature=0.7,
    max_tokens=1024
)

# 定义颜色代码
class Colors:
    USER = '\033[94m'      # 蓝色
    ASSISTANT = '\033[92m'  # 绿色
    RESET = '\033[0m'      # 重置颜色

# 初始化对话历史
conversation_history = []
MAX_HISTORY_TURNS = 5  # 限制对话历史的最大轮次

# 创建一个循环来实现多轮对话
while True:
    # 获取用户输入，使用蓝色显示
    print(f"{Colors.USER}User: {Colors.RESET}", end="")
    prompt = input()
    
    # 检查是否退出
    if prompt.lower() == 'q':
        print("Program ended")
        break
    
    # 将用户输入添加到对话历史
    conversation_history.append({"role": "user", "content": prompt})
    
    # 如果对话历史超过最大轮次，删除最早的对话
    if len(conversation_history) > MAX_HISTORY_TURNS * 2:  # *2是因为每轮包含用户和助手各一条消息
        conversation_history = conversation_history[-MAX_HISTORY_TURNS * 2:]
    
    # 构建完整的对话上下文
    full_prompt = "\n".join([f"{'User' if msg['role']=='user' else 'Assistant'}: {msg['content']}" 
                            for msg in conversation_history])
    
    # 生成回答
    outputs = model.generate([full_prompt], sampling_params)
    
    # 获取回答并清理格式
    response = outputs[0].outputs[0].text.strip()
    # 移除可能的"Assistant:"前缀
    response = response.replace("Assistant:", "").strip()
    
    # 将处理后的回答添加到对话历史
    conversation_history.append({"role": "assistant", "content": response})

    # 使用绿色显示助手回答
    print(f"\n{Colors.ASSISTANT}Assistant: {response}{Colors.RESET}")
    print("\nGenerated tokens:", len(outputs[0].outputs[0].token_ids))
    print("\n" + "="*50 + "\n")