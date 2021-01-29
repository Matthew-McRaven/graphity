import itertools
import functools

import torch
import torch.distributions
import numpy as np

# Implement the hamiltonian discussed with Betre on 20201015
class ASquaredD:
    decomposable = False
    def __init__(self, d, keep_diag=False):
        self.d = d
        self.keep_diag = keep_diag

    @staticmethod
    @functools.lru_cache(1000)
    def compute(adj, d, keep_diag):
        # For each matrix in the batch, compute the adjacency matrix^2.
        temp = torch.matmul(adj, adj) - d
        # Perform element-wise square.
        temp = temp.pow(2)
        # Only mask out diagonal if required.
        if not keep_diag:
            # Construct a matrix only containing diagonal
            n = temp.shape[1]
            diag = (temp.diagonal(dim1=-2,dim2=-1).view(-1) * torch.eye(n, n).view(temp.shape[1:])).long()
            #print(adj, "\n", temp)
            # Subtract out diagonal, so diagonal is 0.
            temp -= diag

        # Sum over the last two dimensions, leaving us with a 1-d array of values.
        # Sum over all non-diagonals.
        return torch.sum(temp, (1,2)) / 2

    # Allow the class to be called like a function.
    def __call__(self, adj, prev_contrib = None, changed_sites = None):
        # Force all tensors to be batched.
        if len(adj.shape) == 2:
            adj = adj.view(1,*adj.shape)
        # At this time, I (MM) don't know how matmul will work in 4+ dims.
        # We will fiure this out when it becomes useful.
        elif len(adj.shape) > 3:
            assert False and "Batched input can have at most 3 dimensions" 
        return self.compute(adj, self.d, self.keep_diag), None

# ASquaredD has explodes in high dimensions.
# Taking the log of this number reduces growth to be more like N**2 rather than a**n**2.
class LogASquaredD(ASquaredD):
    def __init__(self, d, **kwargs):
        super(LogASquaredD, self).__init__(d, **kwargs)
    def __call__(self, adj, prev_contrib = None, changed_sites = None):
        energy, contrib = super(LogASquaredD, self).__call__(adj, prev_contrib, changed_sites)
        return np.log(energy), contrib

# For *really* high dimensions, you may need nested logs.
# However, this will torpedo the ablity to dilineate between states of
# of similiar (but not identical) energy levels.
class NestedLogASquaredD(ASquaredD):
    def __init__(self, d, nesting, **kwargs):
        super(NestedLogASquaredD, self).__init__(d, **kwargs)
        self.nesting = nesting

    def __call__(self, adj, prev_contrib = None, changed_sites = None):
        energy, contrib = super(NestedLogASquaredD, self).__call__(adj, prev_contrib, changed_sites)
        for i in range(self.nesting):
            energy = np.log(energy)
        return energy, contrib