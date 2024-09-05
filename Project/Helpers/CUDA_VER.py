import torch
print(torch.cuda.is_available())  # Should return True if GPU is accessible
print(torch.version.cuda)         # Shows the CUDA version PyTorch is using
