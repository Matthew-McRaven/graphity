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


class FlattenInput(nn.Module):
	def __init__(self, input_dimension):
		super(FlattenInput, self).__init__()
		self.input_dimensions = input_dimension
		self.output_dimension = (functools.reduce(lambda x,y: x*y, input_dimension, 1),)
	def forward(self, input):
		return input.view(-1)