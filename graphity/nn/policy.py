import torch
from torch import dropout, dtype, tensor, tril
import torch.nn as nn
import torch.distributions, torch.nn.init
from torch.nn.modules.loss import MSELoss
import torch.optim

from graphity.hypers import to_cuda

# Policy network internally outputs means of a multivariate distribution that is
# combined with a constant diagonal covariance matrix.
# Previously, I generated a triangular covariance matrix.
# However, it was hard to generate a matrix with a positive diagonal but allow for negative values below the diagonal.
# Rewards with the triangular matrix were low, so I switched to a diagonal covariance matrix.
# This performed slighlty better, but the optimizer wouldn't stop growing the covariance values, leading to eratic behavior.
class MLPPolicyNetwork(nn.Module):
    def __init__(self, input_dimensions, hypers, layers=[28, 14]):
        super(MLPPolicyNetwork, self).__init__()
        # Cache hyperparameters
        self.input_dimensions = input_dimensions
        self.output_pairs = 1
        self.hypers = hypers

        # Build linear layers from input defnition
        linear_layers = []
        previous = input_dimensions
        for index,layer in enumerate(layers):
            linear_layers.append(nn.Linear(previous, layer))
            linear_layers.append(nn.LeakyReLU())
            # We have an extra component at the end, so we can dropout after every layer.
            linear_layers.append(nn.Dropout(hypers['dropout']))
            previous = layer

        self.linear_layers = nn.Sequential(*linear_layers)

        # Use output of NN's
        # Layers that are used to generate our actions using softmax
        self.output_layers = {}
        self.output_layers["first"] = nn.Linear(previous, hypers['graph_size'])
        self.output_layers["second"] = nn.Linear(previous, hypers['graph_size'])
        self.output_layers = nn.ModuleDict(self.output_layers)

        self.softmax = nn.Softmax(dim=0)
        # Initialize NN
        for x in self.parameters():
            if x.dim() > 1:
                nn.init.kaiming_normal_(x)

    def forward(self, input):
        input = input.view(-1)
        # Push observations through feed forward layers.
        output = self.linear_layers(input.float())
        actions = []
        
        # Treat the outputs of my softmaxes as the probability distribution for my NN.
        first_preseed = self.output_layers["first"  ](output)#.clamp(0, 1)
        second_preseed = self.output_layers["second"](output)#.clamp(0, 1)

        #print(torch.max(first_seed))
        first_seed = first_preseed - torch.max(first_preseed)
        second_seed = second_preseed - torch.max(second_preseed)
        first_seed = self.softmax(first_seed)
        second_seed = self.softmax(second_seed)

        if torch.isnan(first_seed).any() or torch.isnan(second_seed).any():
            print(output)
            print(first_preseed, first_seed)
            print(second_preseed, second_seed)
            assert 0
            

        policy = BiCategoricalPolicy(first_seed, second_seed)
        # Sample edge pair to toggle
        actions = policy.sample((self.output_pairs,))

        # Edges are independent of eachother, so joint probability is them multiplied together.
        log_prob = policy.log_prob(actions)
        # Arrange actions so they look like actions from other models.

        return actions, log_prob, policy

class BiCategoricalPolicy:
    def __init__(self, first_seed, second_seed):
        self.first_seed = first_seed
        self.second_seed = second_seed
        # Create integer distributions from my network outputs
        self.first_dist = torch.distributions.Categorical(first_seed)
        self.second_dist = torch.distributions.Categorical(second_seed)

    def log_prob(self, actions):
        return self.first_dist.log_prob(actions[:,0]) * self.second_dist.log_prob(actions[:,1])

    def sample(self, size):
        # Sample edge pair to toggle
        first_sample = self.first_dist.sample(size)
        second_sample = self.first_dist.sample(size)
        return torch.stack([first_sample, second_sample], dim=-1)