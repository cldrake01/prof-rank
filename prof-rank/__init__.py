import torch

# Check available device
if torch.cuda.is_available():
    torch.device("cuda")
    print("GPU is available. Using CUDA.")
elif torch.backends.mps.is_available():
    torch.device("mps")
    print("Apple Silicon is available. Using MPS.")
else:
    torch.device("cpu")
    print("Using CPU.")
