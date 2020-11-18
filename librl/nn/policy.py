import math

import torch
import torch.distributions, torch.nn.init

#########################
#     Policy Objects    #
#########################
# Encapsulate the sampling / logprobs of a policy within a class.
# This helps out the VPG/PPO/PGB code because it doesn't need to know the form
# of the action distribution, it just has to call .log_prob()

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

# Action distribution based on a multivariate-normal distribution.
# Allows actions to be cross-correlated
class MultivariateNormal:
    def __init__(self, mu, lower_tril):
        self.mu = mu
        self.lower_tril = lower_tril
        # Create integer distributions from my network outputs
        self.dist = torch.distributions.MultivariateNormal(mu, scale_tril=self.lower_tril)

    def log_prob(self, actions):
        # Since I'm dealing log probs rather than probs, must add together.
        return self.dist.log_prob(actions)

    def sample(self, size):
        return self.dist.rsample(size)

# Action distribution based on repeated, independent normal distributions.
# Requires that all actions are independent.
class RepeatedNormal:
    def __init__(self, mu, var):
        self.mu = mu
        self.std = var.sqrt()
        # Create integer distributions from my network outputs
        self.dist = torch.distributions.Normal(mu, self.std)

    def log_prob(self, actions):
        # Since I'm dealing log probs rather than probs, must add together.
        return torch.sum(self.dist.log_prob(actions))

    def sample(self, size):
        return self.dist.rsample(size)

# Credit to exisiting "cheating" normal distribution that seems to perform well.
# However, in doesn't seem to do much better than an actual normal distribution in pratice.
# See:
#    https://github.com/andrewliao11/pytorch-a3c-mujoco/blob/master/train.py
class CheatingNormal:
    def __init__(self, mu, var):
        self.mu = mu
        self.var = var

    def log_prob(self, actions):
        pi = torch.FloatTensor([math.pi])
        a = (-1*(actions-self.mu).pow(2)/(2*self.var)).exp()
        b = 1/(2*self.var*pi.expand_as(self.var)).sqrt()
        return -a*b

    def sample(self, size):
        return self.mu + self.var.sqrt()*torch.randn((*size, *self.mu.shape))