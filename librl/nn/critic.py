import functools

import more_itertools
import torch
import torch.nn as nn
import torch.distributions, torch.nn.init
import torch.optim

# Network that learns the expected reward from a state.
class ValueCritic(nn.Module):
    def __init__(self, neural_module, hypers, values=1):
        super(ValueCritic, self).__init__()
        self.neural_module = neural_module
        self.input_dimension = list(more_itertools.always_iterable(neural_module.output_dimension))
        self.__input_size = functools.reduce(lambda x,y: x*y, self.input_dimension, 1)
        self.output_dimension = values
        self.output_layer = torch.nn.Linear(self.__input_size, values)

        for x in self.parameters():
            if x.dim() > 1:
                nn.init.kaiming_normal_(x)

    def forward(self, input):
        output = self.neural_module(input)
        output = self.output_layer(output)
        return output
