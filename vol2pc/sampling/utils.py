import numpy as np
import torch

def to_torch(x, dtype=None, device=None):
    if isinstance(x, torch.Tensor):
        return x.detach().clone().to(device=device, dtype=dtype)
    return torch.tensor(x, device=device, dtype=dtype)