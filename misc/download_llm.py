# 在有网络的环境中
from huggingface_hub import snapshot_download

# 下载模型到本地
model_path = snapshot_download(
    repo_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    local_dir="./deepseek-r1-distill-qwen-7b"  # 指定保存位置
)

print('success')