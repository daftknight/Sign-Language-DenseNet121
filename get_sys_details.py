import torch
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch._C._cuda_getCompiledVersion())
if torch.cuda.is_available():
    print("GPU device:", torch.cuda.get_device_name(0))