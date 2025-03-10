import torch

# 选择GPU 3
device = torch.device("cuda:3")
torch.cuda.set_device(device)

print(f"当前使用的GPU: {torch.cuda.current_device()}")
print(f"CUDA是否可用: {torch.cuda.is_available()}")

# 分配70GB显存
dummy_tensor = torch.zeros((1024 * 1024 * 1024 * 70 // 4,), dtype=torch.float32, device=device)

# 保持程序运行，防止显存被释放
input("按回车键退出程序...")

