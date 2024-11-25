import torch
from models import PoseBridge  # Adjust the import path based on your file structure

# Initialize the model
model = PoseBridge(n_markers=143, marker_dim=3, model_dim=64, num_heads=8, num_layers=4, seq_len=62)

# Calculate total number of trainable parameters
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params}")

# Optional: Detailed breakdown of parameters
print("Parameter breakdown by layer:")
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"{name}: {param.numel()}")
