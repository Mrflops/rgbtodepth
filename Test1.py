import torch

# Check the number of available GPUs
num_gpus = torch.cuda.device_count()

if num_gpus > 0:
    # Use the first available GPU
    device = torch.device("cuda:0")
    print("GPU is available.")
    print("Using GPU:", torch.cuda.get_device_name(device))
else:
    print("GPU is not available.")
    # If no GPU is available, fall back to CPU
    device = torch.device("cpu")

# You can then use this device in your PyTorch code
# For example, if you have a tensor 'x', you can move it to the device with:
# x = x.to(device)
