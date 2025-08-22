import torch

print("Intel GPU available:", torch.xpu.is_available())
print("Device name:", torch.xpu.current_device() if torch.xpu.is_available() else "CPU")

