import torch
import torch.nn as nn
import numpy as np
from numpy.random import Generator, PCG64, random

import librl.agent
# The random agent random selects one edge pair to toggle per timestep.
@librl.agent.add_agent_attr()
class RandomAgent(nn.Module):
    def __init__(self, hypers):
        # Must initialize torch.nn.Module
        super(RandomAgent, self).__init__()
        # I like the PCG RNG, and since we aren't trying to "learn"
        # anything for this agent, numpy's RNGs are fine
        self.rng = Generator(PCG64())

    # Our action is just asking the pytorch implementation for a random set of nodes.
    def act(self, adj, toggles=1):
        return self.forward(adj, toggles)

    # Implement required pytorch interface
    def forward(self, adj, toggles=1):
        # Force all tensors to be batched.
        if len(adj.shape) == 2:
            adj = adj.view(1,*adj.shape)
        # At this time, I (MM) don't know how matmul will work in 4+ dims.
        # We will fiure this out when it becomes useful.
        elif len(adj.shape) > 3:
            assert False and "Batched input can have at most 3 dimensions" 
        # Generate a single pair of random numbers for each adjacency matrix in the batch,
        randoms = self.rng.integers(0, high=adj.shape[-1], size=[toggles, 2])
        # We want to work on tensors, not numpy objects. Respect the device from which the input came.
        randoms = torch.tensor(randoms, device=adj.device)
        # All actions are equally likely, so our chance of choosing this pair is 1/(number of edge pairs) ** (number of edges)
        return randoms, torch.full((1,),1/adj.shape[-1]**(2*toggles)).log()