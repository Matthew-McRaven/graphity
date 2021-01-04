import torch
import torch.nn as nn
import numpy as np
from numpy.random import Generator, PCG64

class random_sampling_strategy:
    def __init__(self, toggles=1, **kwargs):
        self.rng = Generator(PCG64())
        self.toggles = toggles
    def __call__(self, adj):
        # Generate a single pair of random numbers for each adjacency matrix in the batch.
        randoms = self.rng.integers(0, high=adj.shape[-1], size=[self.toggles, 2])
        # We want to work on tensors, not numpy objects. Respect the device from which the input came.
        action = torch.tensor(randoms, device=adj.device)
        # Okay, this term should be more complex, because it ignores the probability delta_e > 0.
        # However, let's assume actions don't have mutual dependence
        logprob = 1/adj.shape[-1]**(2*self.toggles)
        return action, logprob

