import os
import random
import torch
import numpy as np
from torch import nn

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic (slower but reproducible)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"

def save_model(model: nn.Module, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"✓ Saved model to: {path}")

def load_model(model: nn.Module, path: str, map_location: str | None = None):
    model.load_state_dict(torch.load(path, map_location=map_location))
    print(f"✓ Loaded weights from: {path}")

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
