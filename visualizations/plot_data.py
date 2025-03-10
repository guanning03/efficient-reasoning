import matplotlib.pyplot as plt
import json

# 读取数据
with open('results/qwen_sft_gsm8k_batch.json', 'r') as f:
    data = json.load(f)

# 提取数据点
steps = [0, 276, 552, 828, 1104, 1380]
epochs = [0, 1, 2, 3, 4, 5]
train_pass_rates = [data[str(step)]['train']['scores']['average_pass_rate'] for step in steps]
test_pass_rates = [data[str(step)]['test']['scores']['average_pass_rate'] for step in steps]
train_tokens = [data[str(step)]['train']['scores']['avg_tokens'] for step in steps]
test_tokens = [data[str(step)]['test']['scores']['avg_tokens'] for step in steps]

# 创建图形和子图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 3))

# 绘制第一个子图：Pass Rates
ax1.plot(epochs, train_pass_rates, 'b-o', label='Train Pass Rate')
ax1.plot(epochs, test_pass_rates, 'r-o', label='Test Pass Rate')
ax1.set_xlabel('SFT Epochs')
ax1.set_ylabel('Average Pass Rate')
ax1.set_title('Train vs Test Pass Rates')
ax1.grid(True)
ax1.legend()
ax1.set_xticks(epochs)

# 绘制第二个子图：Average Tokens
ax2.plot(epochs, train_tokens, 'b-o', label='Train Avg Tokens')
ax2.plot(epochs, test_tokens, 'r-o', label='Test Avg Tokens')
ax2.set_xlabel('SFT Epochs')
ax2.set_ylabel('Average Tokens')
ax2.set_title('Train vs Test Average Tokens')
ax2.grid(True)
ax2.legend()
ax2.set_xticks(epochs)

# 调整子图之间的间距
plt.tight_layout()

# 保存图片
plt.savefig('training_metrics.png', dpi=300, bbox_inches='tight')
plt.close()