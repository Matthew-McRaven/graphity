import torch
import torch.nn as nn
import torch.distributions, torch.nn.init
import torch.optim

# Network that learns the expected reward from a state.
class MLPCritic(nn.Module):
    def __init__(self, input_dimensions, hypers, layers=[100, 100, 1]):
        super(MLPCritic, self).__init__()
        self.input_dimensions = input_dimensions
        self.output_dimensions = layers[-1]

        # Construct a sequentil linear model from the layer specification.
        linear_layers = []
        previous = input_dimensions
        for index,layer in enumerate(layers):
            linear_layers.append(nn.Linear(previous, layer))
            linear_layers.append(nn.LeakyReLU())
            # Add dropout if this is not the last layer.
            if index < len(layers) - 2:
                linear_layers.append(nn.Dropout(hypers['dropout']))
            previous = layer
        self.linear_layers = nn.Sequential(*linear_layers)

        for x in self.parameters():
            if x.dim() > 1:
                nn.init.kaiming_normal_(x)

    def forward(self, input):
        input = input.view(-1, self.input_dimensions)
        output = self.linear_layers(input.float())
        return output
