import matplotlib.pyplot as plt
import numpy as np

# 数据准备
r1_1_5b = [0.6444, 310.74]      # Pass Rate, Avg Tokens for R1-1.5B
r1_1_5b_rloo = [0.7506, 211.69]  # Pass Rate, Avg Tokens for R1-1.5B-RLOO

# 设置柱子的位置
x = np.arange(2)
width = 0.35

# 创建图形和主坐标轴
fig, ax1 = plt.subplots(figsize=(4, 3))

# 创建第二个Y轴
ax2 = ax1.twinx()

# 绘制柱状图
rects1_pass = ax1.bar(0, r1_1_5b[0], width, label='Pass Rate', color='#2878B5')
rects2_pass = ax1.bar(1, r1_1_5b_rloo[0], width, color='#2878B5')

rects1_tokens = ax2.bar(0 + width, r1_1_5b[1], width, label='Avg Tokens', color='#9AC9DB')
rects2_tokens = ax2.bar(1 + width, r1_1_5b_rloo[1], width, color='#9AC9DB')

# 设置两个Y轴的范围和标签
ax1.set_ylim(0, 1)  # Pass Rate的范围为0-1
ax2.set_ylim(0, 350)  # Avg Tokens的范围为0-2500

ax1.set_ylabel('Pass Rate')
ax2.set_ylabel('Avg Tokens')

# 设置x轴标签
ax1.set_xticks([0.175, 1.175])  # 调整标签位置到两个柱子的中间
ax1.set_xticklabels(['Qwen2.5-1.5B-Ins', 'Qwen2.5-1.5B-RLOO'])

# 修改autolabel函数来正确处理bar对象
def autolabel(rect, ax, is_percentage=True):
    height = rect.get_children()[0].get_height()  # 修正获取高度的方式
    if is_percentage:  # Pass Rate显示百分比
        ax.annotate(f'{height:.1%}',
                    xy=(rect.get_children()[0].get_x() + rect.get_children()[0].get_width()/2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')
    else:  # Avg Tokens显示实际值
        ax.annotate(f'{height:.1f}',
                    xy=(rect.get_children()[0].get_x() + rect.get_children()[0].get_width()/2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')

# 添加标签
autolabel(rects1_pass, ax1, True)
autolabel(rects2_pass, ax1, True)
autolabel(rects1_tokens, ax2, False)
autolabel(rects2_tokens, ax2, False)

plt.tight_layout()

# 保存图片，设置DPI为300以获得高质量图片
plt.savefig('qwen_1.5B_instruct_comparison.pdf', dpi=300, bbox_inches='tight')
plt.close()  # 关闭图形，释放内存