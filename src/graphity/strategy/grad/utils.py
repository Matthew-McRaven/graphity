import torch

def mask_grads(grad):
    size = grad.shape[-1]
    # Prevent NN from toggling diagonal and duplicate edges in upper tril.
    upper_mask = torch.full((size,size), float('inf')).triu(0)
    return -(grad + upper_mask)