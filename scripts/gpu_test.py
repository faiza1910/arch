import torch

print("CUDA available:", torch.cuda.is_available())
print("Number of GPUs:", torch.cuda.device_count())
print("GPU Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")

# Optional: Run a small tensor operation on GPU
if torch.cuda.is_available():
    x = torch.rand(10000, 10000).to('cuda')
    y = x @ x
    print("Operation complete on GPU!")
else:
    print("No GPU found.")
