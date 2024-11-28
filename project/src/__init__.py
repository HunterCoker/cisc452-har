import torch

if __name__ == '__main__':
    if torch.backends.mps.is_available():
        device = torch.device("mps")  # macOS GPU
    elif torch.cuda.is_available():
        device = torch.device("cuda")  # Linux/Windows GPU
    else:
        device = torch.device("cpu")  # Fallback

    print(f"Using device: {device}")