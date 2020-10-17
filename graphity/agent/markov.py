import torch
import torch.nn as nn
from numpy.random import Generator, PCG64

class MDPAgent(nn.Module):
    def __init__(self):
        # Must initialize torch.nn.Module
        super(MDPAgent, self).__init__()
        # I like the PCG RNG, and since we aren't trying to "learn"
        # anything for this agent, numpy's RNGs are fine
        self.rng = Generator(PCG64())

    # Our action is just asking the pytorch implementation for a random set of nodes.
    def act(self, adj):
        return self.forward(adj)

    # Implement required pytorch interface
    def forward(self, adj):
        # Force all tensors to be batched.
        if len(adj.shape) == 2:
            adj = adj.view(1,*adj.shape)
        # At this time, I (MM) don't know how matmul will work in 4+ dims.
        # We will fiure this out when it becomes useful.
        elif len(adj.shape) > 3:
            assert False and "Batched input can have at most 3 dimensions" 
        # Generate a single pair of random numbers for each adjacency matrix in the batch,
        randoms = self.rng.integers(0,high=adj.shape[-1],size=[adj.shape[0],2])
        # We want to work on tensors, not numpy objects. Respect the device from which the input came.
        randoms = torch.tensor(randoms, device=adj.device)
        return randoms