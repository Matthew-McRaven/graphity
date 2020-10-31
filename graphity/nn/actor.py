import torch
from torch import dropout, dtype, tensor, tril
import torch.nn as nn
import torch.distributions, torch.nn.init
from torch.nn.modules.loss import MSELoss
import torch.optim

from graphity.hypers import to_cuda
import graphity.nn.policy

# Agent network based on a submission to my COSC689 class
# It is a stochastic policy network. It will return the policy from forward,
# and you can use this policy to generate further samples
# The current policy is sampling random values from a torch Categorical distrubtion
# conditioned on the output of a linear network.
# The Categorical distribution is non-differentiable, so this may cause
# problems for future programmers.
class MLPActor(nn.Module):
    def __init__(self, input_dimensions, hypers, layers=[28, 14]):
        super(MLPActor, self).__init__()
        
        self.input_dimensions = input_dimensions
        # TODO: toggle more than one edgepair per timestep.
        self.toggles_per_step = hypers['toggles_per_step']
        # Cache hyperparameters locally.
        self.hypers = hypers

        # Build linear layers from input defnition.
        linear_layers = []
        previous = input_dimensions
        for index,layer in enumerate(layers):
            linear_layers.append(nn.Linear(previous, layer))
            linear_layers.append(nn.LeakyReLU())
            # We have an extra component at the end, so we can dropout after every layer.
            linear_layers.append(nn.Dropout(hypers['dropout']))
            previous = layer

        self.linear_layers = nn.Sequential(*linear_layers)

        # Our output layers are used as the seed for some set of random number generators.
        # These random number generators are used to generate edge pairs.
        self.output_layers = {}
        self.output_layers["first"] = nn.Linear(previous, hypers['graph_size'])
        self.output_layers["second"] = nn.Linear(previous, hypers['graph_size'])
        self.output_layers = nn.ModuleDict(self.output_layers)

        # Must pass output layers through softmax in order for them to be a proper PDF.
        self.softmax = nn.Softmax(dim=0)

        # Initialize NN
        for x in self.parameters():
            if x.dim() > 1:
                nn.init.kaiming_normal_(x)

    def forward(self, input):
        # TODO: Handle batched inputs
        input = input.view(self.input_dimensions)
        # Push observations through feed forward layers.
        output = self.linear_layers(input.float())
        actions = []

        # When graph dimension becomes large, outputs of linear layers can underflow and overflow.
        # If this happens, you need to shrink the dimension of the graph (bad), reduce the learning rate,
        # decrease the magnitude of the loss (do this by changing the reward function to log(reward function))
        # or widening the datatypes of our networks.
        # 2020-10-30 (MM) -- encountered overflow with sum((A^2-d)^2) at n=100. Fix by taking log(expression) as
        # new energy function.
        if torch.isnan(output).any():
            print(output)
            assert 0 and "Output overflowed from neural net. Abandon all hope ye who enter here."

        # Treat the outputs of my softmaxes as the probability distribution for my NN.
        first_preseed = self.output_layers["first"  ](output)
        second_preseed = self.output_layers["second"](output)

        # Since softmax will rescale all numbers to sum to 1,
        # logically it doesn't matter where the sequence lies on the number line.
        # Since softmax will compute e(x), I want all x's to be small.
        # To do this, I subtract the maximum value from every element, moving my
        # numbers from (-∞,∞) to (-∞,0]. This makes softmax more stable
        first_seed = first_preseed - torch.max(first_preseed)
        second_seed = second_preseed - torch.max(second_preseed)
        first_seed = self.softmax(first_seed)
        second_seed = self.softmax(second_seed)

        # Encapsulate our poliy in an object so downstream classes don't
        # need to know what kind of distribution to re-create.
        policy = graphity.nn.policy.BiCategoricalPolicy(first_seed, second_seed)
        # Sample edge pair to toggle
        actions = policy.sample((self.toggles_per_step,))
        # Each actions is drawn independtly of others, so joint prob
        # is all of them multiplied together. However, since we have logprobs,
        # we need to sum instead.
        log_prob = torch.sum(policy.log_prob(actions))
        # Arrange actions so they look like actions from other models.

        return actions, log_prob, policy