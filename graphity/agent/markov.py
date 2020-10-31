import torch
import torch.nn as nn
import numpy as np
from numpy.random import Generator, PCG64, random

# Needed for add_agent_attr() decorator
import graphity.agent

# The random agent random selects one edge pair to toggle per timestep.
@graphity.agent.add_agent_attr()
class RandomAgent(nn.Module):
    def __init__(self, hypers):
        # Must initialize torch.nn.Module
        super(RandomAgent, self).__init__()
        # I like the PCG RNG, and since we aren't trying to "learn"
        # anything for this agent, numpy's RNGs are fine
        self.rng = Generator(PCG64())
        self.toggles_per_step = hypers['toggles_per_step']

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
        randoms = self.rng.integers(0, high=adj.shape[-1],size=[self.toggles_per_step,2])
        # We want to work on tensors, not numpy objects. Respect the device from which the input came.
        randoms = torch.tensor(randoms, device=adj.device)
        # All actions are equally likely, so our chance of choosing this pair is 1/(number of edge pairs) ** (number of edges)
        return randoms, torch.full((1,),1/adj.shape[-1]**(2*self.toggles_per_step)).log()

# Markov agent is willing to back out last edge, with some prbability, if that action increased the energy of the sytstem.
# This "regret" factor is beta, the inverse of the temperature.
@graphity.agent.add_agent_attr(allow_callback=True)
class MDPAgent(nn.Module):
    def __init__(self,  hypers, beta=2):
        # Must initialize torch.nn.Module
        super(MDPAgent, self).__init__()
        # I like the PCG RNG, and since we aren't trying to "learn"
        # anything for this agent, numpy's RNGs are fine
        self.rng = Generator(PCG64())
        self.toggles_per_step = hypers['toggles_per_step']
        # Multipliciative inverse of the temperature of the system.
        self.beta = beta
        # Last two (action, reward pairs)
        self.arm2 = None
        self.arm1 = None
        # Last action taken by NN.
        self.am1 = None

    # Our action is just asking the pytorch implementation for a random set of nodes.
    def act(self, adj):
        return self.forward(adj)

    # Callback, which will allow us to "undo" our last action if it is worse.
    def act_callback(self, _, reward):
        self.arm2 = self.arm1
        # Join internally cached action and environment-supplied reward
        self.arm1 = (self.am1, reward)

    # Generate a random adjacency flip on matrix adj.
    def random(self, adj):
        # Generate a single pair of random numbers for each adjacency matrix in the batch.
        randoms = self.rng.integers(0, high=adj.shape[-1],size=[self.toggles_per_step,2])
        # We want to work on tensors, not numpy objects. Respect the device from which the input came.
        randoms = torch.tensor(randoms, device=adj.device)
        return randoms

    # Implement required pytorch interface
    def forward(self, adj):
        # Force all tensors to be batched.
        if len(adj.shape) == 2:
            adj = adj.view(1,*adj.shape)
        # At this time, I (MM) don't know how matmul will work in 4+ dims.
        # We will fiure this out when it becomes useful.
        elif len(adj.shape) > 3:
            assert False and "Batched input can have at most 3 dimensions" 

        # Need to cache which action weare going to take
        action, log_prob = None, None
        # If we do not have two look-back states, we can't make inferences about environment.
        if not self.arm1 or not self.arm2:
            action = self.random(adj)
            # Not enough info to back out an action, so our probability of choosing
            # an action is just the probability of choosing a random edge pair.
            log_prob = torch.full((1,),1/adj.shape[-1]**(2*self.toggles_per_step)).log()
        else:
            # Otherwise, we need to compute how much system energy changed.
            delta_e = self.arm1[1] - self.arm2[1]
            random_number = self.rng.random()
            #print(f"current:{self.arm1[1]}, old:{self.arm2[1]}, ΔE:{delta_e}")
            # If we gained energy because of the last step
            if delta_e > 0:
                #print(f"CPW {random_number}, {np.exp(-(self.beta*delta_e))}")
                # Roll a random in [0., 1.], and compare against a function of system temperature and energy.
                # If we "lose", back out our last action that increased system energy.
                # Otherwise, accept the action and randomly create flip an edge. 
                to_beat = np.exp(-(self.beta*delta_e))
                if random_number < to_beat:
                    action = self.arm1[0]
                    # The probability of this action is the probability of us failing the roll.
                    log_prob = torch.full((1,), to_beat.item()).log()
                else:
                    action = self.random(adj)
                    # The probability of this action is the probability of beating the roll
                    # and randomly choosing this edge.
                    log_prob = ((1-to_beat)*torch.full((1,),1/adj.shape[-1]**(2*self.toggles_per_step))).log()
            else:
                action = self.random(adj)
                # Okay, this term should be more complex, because it ignores the probability delta_e > 0.
                # However, let's assume actions dn't have mutual dependence
                log_prob = torch.full((1,),1/adj.shape[-1]**(2*self.toggles_per_step)).log()
        # Must cache last action here, because it isn't available in act_callback(...)
        self.am1 = action
        return action, log_prob
            


