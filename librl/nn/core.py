import functools

import more_itertools
import torch
import torch.nn as nn
import torch.optim

# Agent network based on a submission to my COSC689 class
# It is a stochastic policy network. It will return the policy from forward,
# and you can use this policy to generate further samples
# The current policy is sampling random values from a torch Categorical distrubtion
# conditioned on the output of a linear network.
# The Categorical distribution is non-differentiable, so this may cause
# problems for future programmers.
class MLPKernel(nn.Module):
    def __init__(self, input_dimensions, layer_list=[200, 100], dropout=.1):
        super(MLPKernel, self).__init__()
        
        self.input_dimensions = list(more_itertools.always_iterable(input_dimensions))

        # Build linear layers from input defnition.
        linear_layers = []
        previous = functools.reduce(lambda x, y: x*y, self.input_dimensions, 1)
        for index,layer in enumerate(layer_list):
            linear_layers.append(nn.Linear(previous, layer))
            linear_layers.append(nn.LeakyReLU())
            # We have an extra component at the end, so we can dropout after every layer.
            linear_layers.append(nn.Dropout(dropout))
            previous = layer
        self.output_dimension = (previous, )
        self.linear_layers = nn.Sequential(*linear_layers)

        # Initialize NN
        for x in self.parameters():
            if x.dim() > 1:
                nn.init.kaiming_normal_(x)

    def forward(self, input):
        input = input.view(-1, functools.reduce(lambda x, y: x*y, self.input_dimensions, 1))
        # Push observations through feed forward layers.
        output = self.linear_layers(input.float())

        assert not torch.isnan(output).any()

        return output