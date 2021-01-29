import functools

import more_itertools
import torch
import torch.nn as nn
import torch.distributions, torch.nn.init
import torch.optim


import librl.nn.policy

# Agent network based on a submission to my COSC689 class
# It is a stochastic policy network. It will return the policy from forward,
# and you can use this policy to generate further samples
# The current policy is sampling random values from a torch Categorical distrubtion
# conditioned on the output of a linear network.
# The Categorical distribution is non-differentiable, so this may cause
# problems for future programmers.
class BiCategoricalActor(nn.Module):
    def __init__(self, neural_module, action_space, observation_space):
        super(BiCategoricalActor, self).__init__()
        
        self.input_dimension = list(more_itertools.always_iterable(neural_module.output_dimension))
        self._input_size = functools.reduce(lambda x,y: x*y, self.input_dimension, 1)
        self.neural_module = neural_module
        self.output_dimension = 2
        assert len (action_space.shape) == 2 and action_space.shape[-1] == 2
        assert len (observation_space.shape) == 2 and observation_space.shape[0] == observation_space.shape[1]

        # Our output layers are used as the seed for some set of random number generators.
        # These random number generators are used to generate edge pairs.
        self.output_layers = {}
        self.output_layers["first"] = nn.Linear(self._input_size, observation_space.shape[0])
        self.output_layers["second"] = nn.Linear(self._input_size, observation_space.shape[0])
        self.output_layers = nn.ModuleDict(self.output_layers)

        # Must pass output layers through softmax in order for them to be a proper PDF.
        self.softmax = nn.Softmax(dim=0)

        # Initialize NN
        for x in self.parameters():
            if x.dim() > 1:
                nn.init.kaiming_normal_(x)
                
    def recurrent(self):
        return self.neural_module.recurrent()

    def save_hidden(self):
        assert self.recurrent()
        self.neural_module.save_hidden()

    def restore_hidden(self, state=None):
        assert self.recurrent()
        self.neural_module.restore_hidden(state)

    def forward(self, input):
        output = self.neural_module(input).view(-1, self._input_size)
        actions = []

        assert not torch.isnan(output).any() # type: ignore

        # Treat the outputs of my softmaxes as the probability distribution for my NN.
        first_preseed = self.output_layers["first"  ](output).view(-1)
        second_preseed = self.output_layers["second"](output).view(-1)

        # Since softmax will rescale all numbers to sum to 1,
        # logically it doesn't matter where the sequence lies on the number line.
        # Since softmax will compute e(x), I want all x's to be small.
        # To do this, I subtract the maximum value from every element, moving my
        # numbers from (-∞,∞) to (-∞,0]. This makes softmax more stable
        first_seed = first_preseed - torch.max(first_preseed) # type: ignore
        second_seed = second_preseed - torch.max(second_preseed) # type: ignore
        first_seed = self.softmax(first_seed)
        second_seed = self.softmax(second_seed)
        
        # Encapsulate our poliy in an object so downstream classes don't
        # need to know what kind of distribution to re-create.
        policy = librl.nn.policy.BiCategoricalPolicy(first_seed, second_seed)
        # Sample edge pair to toggle
        actions = policy.sample((1,))
        # Each actions is drawn independtly of others, so joint prob
        # is all of them multiplied together. However, since we have logprobs,
        # we need to sum instead.
        log_prob = torch.sum(policy.log_prob(actions)) # type: ignore
        # Arrange actions so they look like actions from other models.

        return actions, log_prob, policy
