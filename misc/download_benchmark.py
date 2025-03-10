from datasets import load_dataset

# dataset1 = load_dataset("Maxwell-Jia/AIME_2024")
# dataset1.save_to_disk("benchmarks/AIME2024")
# dataset2 = load_dataset("opencompass/AIME2025")
# dataset2.save_to_disk("benchmarks/AIME2025")
# 添加 GSM8K 数据集的下载
gsm8k = load_dataset("gsm8k", "main")
gsm8k.save_to_disk("benchmarks/gsm8k")

# 打印数据集信息
print("\n数据集基本信息：")
print(gsm8k)

# 打印训练集中的一个样本
print("\n训练集样本示例：")
sample = gsm8k['train'][0]
print("问题:", sample['question'])
print("\n答案:", sample['answer'])

# 打印测试集中的一个样本
print("\n测试集样本示例：")
sample = gsm8k['test'][0]
print("问题:", sample['question'])
print("\n答案:", sample['answer'])