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
    def __init__(self, annealing_strategy, site_strategy):
        """
        :param annealing_strategy: A function of self.beta and a change in energy.  
        It returns a new beta for the environment and the probability of selecting that beta.
        :param site_strategy: A function of lattices/graphs. 
        It returns a site whose value is to be changed and the probability of selecting that site.
        """
        # Must initialize torch.nn.Module
        super(ForwardAgent, self).__init__()
        self.annealing_strategy = annealing_strategy
        self.site_strategy = site_strategy

    def end_sweep(self): 
        """
        Inform the annealing strategy that time has move forward.
        Must be called after every sweep for adaptive strategies to work correctly.
        """
        self.annealing_strategy.step()

    def end_epoch(self): 
        """
        Inform the annealing strategy that an epoch has ended.
        Must be called so that the annealing strategy can reset itself to starting position for the next epoch.
        """
        self.annealing_strategy.reset()

    def act(self, lattice, delta_e):
        """
        Ask the agent which site should be toggled and which temperature the change should be performed at,
        Returns the log prob of choosing the site+temperature combo.

        :param obj: A lattice (or gaph) which is to be modified.
        :param delta_e: The change in energy from the last change to the change befor it.
        May inform adaptive annealing strategies.
        """
        return self.forward(lattice, delta_e)

    # Implement required pytorch interface
    def forward(self, obj, delta_e):
        """
        Ask the agent which site should be toggled and which temperature the change should be performed at,
        Returns the log prob of choosing the site+temperature combo.

        :param obj: A lattice (or gaph) which is to be modified.
        :param delta_e: The change in energy from the last change to the change befor it.
        May inform adaptive annealing strategies.
        """
        site, lp_action = self.site_strategy(obj)
        beta, lp_beta = self.annealing_strategy(delta_e)
        return (site, beta), lp_action + lp_beta