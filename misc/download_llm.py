# 在有网络的环境中
from huggingface_hub import snapshot_download

# 下载模型到本地
model_path = snapshot_download(
    repo_id="Qwen/Qwen2.5-0.5B",
    local_dir="./Qwen2.5-0.5B"  # 指定保存位置
)

print('success')