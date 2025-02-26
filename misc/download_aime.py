from datasets import load_dataset

dataset1 = load_dataset("Maxwell-Jia/AIME_2024")
dataset1.save_to_disk("benchmarks/AIME2024")
dataset2 = load_dataset("opencompass/AIME2025")
dataset2.save_to_disk("benchmarks/AIME2025")