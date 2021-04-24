import numpy as np
from numpy.random import default_rng
import torch.tensor

def random_graph(graph_size, rng=None):
    if rng is None:
        # Need to generate a random nXn graph that is square, symmetric, integer-valued, and all 1's or 0's.
        rng = default_rng()
    # RNG excludes hi endpoint.
    rand = rng.integers(0,2, (graph_size, graph_size))

    return torch.tensor(rand)

def random_adj_matrix(graph_size, allow_self_loops=False, rng=None):
    if rng is None:
        # Need to generate a random nXn graph that is square, symmetric, integer-valued, and all 1's or 0's.
        rng = default_rng()
    # RNG excludes hi endpoint.
    rand = random_graph(graph_size, rng)
    # Mask out all values on or above the diagonal
    lower_tril = np.tril(rand, -1)
    # TODO: Explain magic of why transpose works.
    upper_tril = lower_tril.transpose()
    # Force the diagonal to be 0 if we disallow self loops.
    diag = np.diag(np.diag(rand)) if allow_self_loops else 0
    # Recombine all tree matrix parts.
    output =  diag + lower_tril + upper_tril
    return torch.tensor(output)