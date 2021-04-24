import functools

import torch.nn as nn
import torch
import numpy as np
# Massage something that looks like a tensor, ndarray to a tensor on the correct device.
# If given an iterable of those things, recurssively attempt to massage them into tensors.
def torchize(maybe_tensor, device):
    # Sometimes an object is a torch tensor that just needs to be moved to the right device
    if torch.is_tensor(maybe_tensor): return maybe_tensor.to(device)
    # Sometimes it is an ndarray
    elif isinstance(maybe_tensor, np.ndarray): return torch.tensor(maybe_tensor).to(device) # type: ignore
    # Maybe a simple iterable of things we need to move to torch, like a cartesian product of ndarrays.
    elif isinstance(maybe_tensor, (list, tuple)): return [torchize(item, device) for item in maybe_tensor]
    elif maybe_tensor is None: return None
    else: raise NotImplementedError(f"Don't understand the datatype of {type(maybe_tensor)}")

# A tensor must have 2 dims to be considered a matrix.
def is_matrix(tensor):
    return len(tensor.shape) == 2

# If we have a tensor of matricies, the last two dims correspond to mXn of the matrix.
# For the tensor to be square, m==n.
def is_square(tensor):
    return tensor.shape[-1] == tensor.shape[-2]

# Return if all matricies in the tensor are symmetric.
def is_symmetric(tensor):
    # Maybe batching has has given us more than 3 dims. If so, flatten it.
    tensor = tensor.view(-1, tensor.shape[-1], tensor.shape[-2])
    # iterate over all matricies, and all indicies
    for k in range(tensor.shape[0]):
        for i in range(tensor.shape[1]):
            # We can start at i+1 rather than 0 because if j were less than i,
            # we would already have check that index in a previous iteration over i.
            # No need to start at i, since diagonal can't affect symmetry.
            for j in range(i+1, tensor.shape[2]):
                if tensor[k,i,j] != tensor [k,j,i]:
                    # Abort early, saving us runtime.
                    return False
    # Proved for all pairs <i, j> where i<j, G[i,j]=G[k,i], i.e. graph is symmetric.
    return True

# Check if a tensor contains only the values 0 or 1, a requirement for adjacency matricies.
def all_zero_one(tensor):
    # Flatten to 1d for ease
    tensor = tensor.view(-1)
    # Perform elementwise equality to int(0) or int (1). This yields a tensor.
    # Then, perform a reduction that and's together all elements and returns a single element tensor.
    # Extrat the truthy value from tensor with .item()
    x = torch.all(torch.eq(tensor, 1) | torch.eq(tensor, 0)).item()
    return x

# A tensor can be considered a adjacency matrix if it has 2 dimensions
# of the same size. Additionally, it must be symmetric with all entries in {0, 1}
def is_adj_matrix(tensor):
    return (is_matrix(tensor) and is_square(tensor) 
            and is_symmetric(tensor) and all_zero_one(tensor))
class FlattenInput(nn.Module):
    def __init__(self, input_dimension):
        super(FlattenInput, self).__init__()
        self.input_dimensions = input_dimension
        self.output_dimension = (functools.reduce(lambda x,y: x*y, input_dimension, 1),)
    def forward(self, input):
        return input.view(-1)