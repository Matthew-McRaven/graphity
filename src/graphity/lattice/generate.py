import numpy as np
from numpy.random import default_rng
import torch.tensor
def random_glass(glass_shape, rng=None):
    if rng is None:
        # Need to generate a random nXn graph that is square, symmetric, integer-valued, and all 1's or 0's.
        rng = default_rng()
    # RNG excludes hi endpoint.
    return torch.tensor(rng.integers(-1,2, glass_shape))
    