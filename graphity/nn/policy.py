import torch
import torch.distributions, torch.nn.init
from torch.nn.modules.loss import MSELoss

class BiCategoricalPolicy:
    def __init__(self, first_seed, second_seed):
        self.first_seed = first_seed
        self.second_seed = second_seed
        # Create integer distributions from my network outputs
        self.first_dist = torch.distributions.Categorical(first_seed)
        self.second_dist = torch.distributions.Categorical(second_seed)

    def log_prob(self, actions):
        # Since I'm dealing log probs rather than probs, must add together.
        return self.first_dist.log_prob(actions[:,0]) + self.second_dist.log_prob(actions[:,1])

    def sample(self, size):
        # Sample edge pair to toggle
        first_sample = self.first_dist.sample(size)
        second_sample = self.first_dist.sample(size)
        return torch.stack([first_sample, second_sample], dim=-1)