import torch
import torch.nn as nn
import numpy as np
from numpy.random import Generator, PCG64

# Needed for add_agent_attr() decorator
import librl.agent
import librl.train.cc.pg 

# An agent without backtracking.
@librl.agent.add_agent_attr()
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

# Markov agent is willing to back out last edge, with some prbability, if that action increased the energy of the sytstem.
# This "regret" factor is beta, the inverse of the temperature.
@librl.agent.add_agent_attr(allow_callback=True)
class MetropolisAgent(nn.Module):
    def __init__(self, sampling_strategy, beta=2):
        # Must initialize torch.nn.Module
        super(MetropolisAgent, self).__init__()
        # I like the PCG RNG, and since we aren't trying to "learn"
        # anything for this agent, numpy's RNGs are fine
        
        # Multipliciative inverse of the temperature of the system.
        self.sampling_strategy = sampling_strategy
        self.beta = beta

        self.rng = Generator(PCG64())
        # Last two (action, reward pairs)
        self.arm2 = None
        self.arm1 = None
        # Last action taken by NN.
        self.am1 = None

    # Our action is just asking the pytorch implementation for a random set of nodes.
    def act(self, adj, toggles=1):
        return self.forward(adj, toggles)

    # Callback, which will allow us to "undo" our last action if it is worse.
    def act_callback(self, reward=None, **_):
        self.arm2 = self.arm1
        # Join internally cached action and environment-supplied reward
        self.arm1 = (self.am1, reward)

    def anneal(self, adj):
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
                # The probability of this action is the probability of us failing the roll.
                return self.arm1[0], to_beat.log()
            else:
                # The probability of this action is the probability of beating the roll
                # and randomly choosing this edge.
                action, log_prob = self.sampling_strategy(adj)
                log_prob += (1-to_beat).log()
                return action, log_prob
        else:
            return self.sampling_strategy(adj)

    # Implement required pytorch interface
    def forward(self, adj, toggles=1):
        # Don't yet deal with batched input.
        assert len(adj.shape) == 2

        # Need to cache which action weare going to take
        action, log_prob = None, None
        # If we do not have two look-back states, we can't make inferences about environment.
        if not self.arm1 or not self.arm2:
            action, log_prob = self.sampling_strategy(adj)
        else:
            action, log_prob = self.anneal(adj)
        # Must cache last action here, because it isn't available in act_callback(...)
        self.am1 = action
        return action, log_prob

# Implements Simulated Annealing.
# See: On the Design of an Adaptive Simulated Annealing Algorithm, Cicirello 2009.
#   https://www.cicirello.org/publications/CP2007-Autonomous-Search-Workshop.pdf
@librl.agent.add_agent_attr(allow_callback=True)
class SimulatedAnnealingAgent(MetropolisAgent):
    def __init__(self, sampling_strategy, alpha, round_length, T0):
        super(SimulatedAnnealingAgent, self).__init__(sampling_strategy)
        self.sampling_strategy = sampling_strategy
        self._timestep = 0
        self.alpha = alpha
        self.T0 = T0
        self.round_length = round_length

    def end_epoch(self): self._epoch = self._timestep = 0

    # Implement required pytorch interface
    def forward(self, adj, toggles=1):
        # Cool the system slightly every timestep.
        self.beta = self.T0 / self.alpha ** np.floor(self._timestep/self.round_length)
        self._timestep += 1
        return super().forward(adj, toggles)
        

# Implement training procedure for grad descent only.
# Skip MAML, because this agent doesn't learning anything. 
@librl.train.cc.pg.policy_gradient_update.register(ForwardAgent)
def _(agent, tasks_iterable):
    pass

# Implement training procedure for grad descent only.
# Skip MAML, because this agent doesn't learning anything. 
@librl.train.cc.pg.policy_gradient_update.register(MetropolisAgent)
def _(agent, tasks_iterable):
    pass