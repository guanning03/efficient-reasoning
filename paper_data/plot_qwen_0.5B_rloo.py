import matplotlib.pyplot as plt
import numpy as np

# 数据准备
labels = ['sft', 'rollout_8', 'rollout_64', 'rollout_512', 'rollout_4096']
pass_rate = [0.3093, 0.3093, 0.2661, 0.2234, 0.1804]

# 创建图形
fig, ax = plt.subplots(figsize=(4, 3))

# 绘制柱状图
bars = ax.bar(range(len(labels)), pass_rate, color='#2878B5')

# 设置y轴范围从0开始
ax.set_ylim(0, max(pass_rate) * 1.2)

# 设置x轴标签
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, rotation=45)

# 设置y轴标签
ax.set_ylabel('Pass Rate')

# 添加数值标签
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.1%}',
                    xy=(rect.get_x() + rect.get_width()/2, height),
                    xytext=(0, 3),  # 3点垂直偏移
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(bars)

# 调整布局
plt.tight_layout()

# 保存图片
plt.savefig('qwen_0.5B_rloo.pdf', dpi=300, bbox_inches='tight')
plt.close()




