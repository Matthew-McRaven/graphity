import torch
import torch.nn as nn
import numpy as np
from numpy.random import Generator, PCG64

# Needed for add_agent_attr() decorator
from . import add_agent_attr

# An agent without backtracking.
@add_agent_attr()
class ForwardAgent(nn.Module):
    """
    A deterministic agent without backtracking.
    This agent is incapable of learning, and will blindly execute its given annealing and site strategies.

 
    Afte a sweep has been completed, end_sweep() must be 
    """
    def __init__(self, annealing_strategy, site_strategy, beta=1.0):
        """
        :param annealing_strategy: A function of self.beta and a change in energy. 
        """
        # Must initialize torch.nn.Module
        super(ForwardAgent, self).__init__()
        self.annealing_strategy = annealing_strategy
        self.beta = beta
        self.site_strategy = site_strategy
    def end_sweep(self): self.annealing_strategy.step()
    def end_epoch(self): self.annealing_strategy.reset()
    def act(self, lattice, delta_e):
        return self.forward(lattice, delta_e)

    # Implement required pytorch interface
    def forward(self, lattice, delta_e):
        site, lp_action = self.site_strategy(lattice)
        beta, lp_beta = self.annealing_strategy(self.beta, delta_e)
        return (site, beta), lp_action + lp_beta