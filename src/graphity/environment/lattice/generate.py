import numpy as np
from numpy.random import default_rng
import torch.tensor


def random_glass(glass_shape, rng=None):
    """
    Create a random spin glass of a given size.

    :param glass_shape: The required dimensions of the output spin glass.
    :param rng: An optional random number generator. 
    Providing one allows you to get deterministic results out of this function via seeded RNGs.
    Defaults to the numpy's default_rng().
    """
    if rng is None:
        # Need to generate a random nXn graph that is square, symmetric, integer-valued, and all 1's or 0's.
        rng = default_rng()
    # RNG excludes hi endpoint.
    #return torch.tensor(rng.integers(-1,2, glass_shape))
    # Rescale {0, 1} to {-1, +1} with some clever math.
    return torch.tensor(rng.integers(0,2, glass_shape)) * 2 -1
    