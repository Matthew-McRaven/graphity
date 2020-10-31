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
    def __init__(self, H, hypers):
        # Must initialize torch.nn.Module
        super(GradientFollowingAgent, self).__init__()
        # I like the PCG RNG, and since we aren't trying to "learn"
        # anything for this agent, numpy's RNGs are fine
        self.rng = Generator(PCG64())
        self.H = H
        self.toggles_per_step = hypers['toggles_per_step']

    def act(self, adj):
        return self.forward(adj)

    # Implement required pytorch interface
    def forward(self, adj):
        size = adj.shape[-1]
        grad = graphity.grad.graph_gradient(adj, self.H).tril()

        # Prevent NN from toggling diagonal and duplicate edges in upper tril.
        upper_inf = torch.full((size,size), float('inf')).triu(0)
        grad = (grad + upper_inf).view(-1)

        # Pick k transitions than minimize energy the most.
        _, indicies = torch.topk(grad, self.toggles_per_step, largest=False)

        # Contains all of the columns
        first = indicies // (size)
        # Contains all of the rows
        second = indicies % (size)
        
        # Must stack along dim=-1 in order to properly join pairs.
        actions = torch.stack([first,second], dim=-1)

        return actions, torch.zeros((1,))