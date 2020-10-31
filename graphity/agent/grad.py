import torch
import torch.nn as nn
import numpy as np
from numpy.random import Generator, PCG64, random

# Needed for add_agent_attr() decorator
import graphity.agent
import graphity.grad

# The random agent random selects one edge pair to toggle per timestep.
@graphity.agent.add_agent_attr()
class GradientFollowingAgent(nn.Module):
    def __init__(self, H):
        # Must initialize torch.nn.Module
        super(GradientFollowingAgent, self).__init__()
        # I like the PCG RNG, and since we aren't trying to "learn"
        # anything for this agent, numpy's RNGs are fine
        self.rng = Generator(PCG64())
        self.H = H

    def act(self, adj):
        return self.forward(adj)

    # Implement required pytorch interface
    def forward(self, adj):
        size = adj.shape[-1]
        grad = graphity.grad.graph_gradient(adj, self.H).view(-1)
        min,idx = torch.min(grad, dim=0)
        #print(min, idx, adj)
        x, y = idx//size, idx%size
        actions = torch.tensor([[x,y]], device=adj.device, dtype=torch.uint8)
        #print(actions)
        # Our probability of choosing the current action is 1, because we only choose the best option.
        # Therefore log prob = 0
        return actions, torch.zeros((1,))