import torch
import torch.nn as nn
import numpy as np
from numpy.random import Generator, PCG64

class RandomSearch:
    def __init__(self, toggles=1, **kwargs):
        self.rng = Generator(PCG64())
        self.toggles = toggles
    def __call__(self, adj):
        h = adj.shape[-1]
        randoms = self.rng.integers((0,0,2), high=(h,h,3), size=[3])
        # We want to work on tensors, not numpy objects. Respect the device from which the input came.
        action = torch.tensor(randoms, device=adj.device)
        # Okay, this term should be more complex, because it ignores the probability delta_e > 0.
        # However, let's assume actions don't have mutual dependence
        logprob = float('nan') #1/adj.shape[-1]**(2*self.toggles)
        return action, logprob


