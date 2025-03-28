import torch

# 查看显卡名称
print("显卡型号:", torch.cuda.get_device_name(0))

# 查看计算能力（架构版本）
print("计算能力:", torch.cuda.get_device_capability(0))   