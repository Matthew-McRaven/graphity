import torch
import torch.nn as nn
import numpy as np
from numpy.random import Generator, PCG64

# Needed for add_agent_attr() decorator
from . import add_agent_attr

# An agent without backtracking.
@add_agent_attr()
class ForwardAgent(nn.Module):
    def __init__(self, sampling_strategy):
        # Must initialize torch.nn.Module
        super(ForwardAgent, self).__init__()
        self.sampling_strategy = sampling_strategy

    def act(self, adj):
        return self.forward(adj)

    # Implement required pytorch interface
    def forward(self, adj):
        # Don't yet deal with batched input.
        assert len(adj.shape) == 2
        action, log_probs = self.sampling_strategy(adj)

        return action, log_probs