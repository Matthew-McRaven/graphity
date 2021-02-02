import torch

def mask_grads(grad):
    # Prevent NN from toggling diagonal and duplicate edges in upper tril.
    upper_mask = torch.full(grad.shape, float('inf')).triu(0)
    return -(grad + upper_mask)