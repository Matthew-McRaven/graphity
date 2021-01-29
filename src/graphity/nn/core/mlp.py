import functools
import types

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
    def __init__(self, input_dimensions, layer_list=None, dropout=None):
        
        super(MLPKernel, self).__init__()
        dropout = dropout if dropout else self.get_default_hyperparameters().dropout
        layer_list = layer_list if layer_list else self.get_default_hyperparameters().layer_list
        self.input_dimensions = list(more_itertools.always_iterable(input_dimensions))
        self._input_size = functools.reduce(lambda x, y: x*y, self.input_dimensions, 1)

        # Build linear layers from input defnition.
        linear_layers = []
        previous = self._input_size
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

    def recurrent(self):
        return False
        
    @staticmethod
    def get_default_hyperparameters():
        ret = {}
        ret['dropout'] = .1
        ret['layer_list'] = [200, 100]
        return types.SimpleNamespace(**ret)

    def forward(self, input):
        input = input.view(-1, self._input_size)
        # Push observations through feed forward layers.
        output = self.linear_layers(input.float())

        assert not torch.isnan(output).any() # type: ignore

        return output