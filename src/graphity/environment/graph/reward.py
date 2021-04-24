import itertools
import functools

import torch
import torch.distributions
import numpy as np

def contribution(adj, d, keep_diag):
    # For each matrix in the batch, compute the adjacency matrix^2.
    temp = torch.matmul(adj, adj) - d
    # Perform element-wise square.
    temp = temp.pow(2)
    # Only mask out diagonal if required.
    if not keep_diag:
        # Construct a matrix only containing diagonal
        n = temp.shape[1]
        diag = temp.diagonal(dim1=-2,dim2=-1) * torch.eye(n, n)
        #print(adj, "\n", temp)
        # Subtract out diagonal, so diagonal is 0.
        temp -= diag
    return temp/2

def compute_betre(adj, d, keep_diag):
    contrib = contribution(adj, d, keep_diag)
    # Sum over contributions, leaving us with a scalar energy.
    return torch.sum(contrib, (1,2)), contrib

# Implement the hamiltonian discussed with Betre on 20201015
class ASquaredD:
    decomposable = False
    def __init__(self, d, keep_diag=False):
        self.d = d
        self.keep_diag = keep_diag
    def contribution(self, adj):
        return contribution(adj, self.d, self.keep_diag)
    # Signature-compatible with lattice hamiltonians.
    def __call__(self, adj, prev_contribs=None, changed_sites=None):
        # Force all tensors to be batched.
        if len(adj.shape) == 2:
            adj = adj.view(1,*adj.shape)
        # At this time, I (MM) don't know how matmul will work in 4+ dims.
        # We will fiure this out when it becomes useful.
        elif len(adj.shape) > 3:
            assert False and "Batched input can have at most 3 dimensions" 
        # If we have no cached information from the previous timestep, we must perform full recompute
        if prev_contribs is None or changed_sites is None:
            energy, contribs =  compute_betre(adj, self.d, self.keep_diag)
            return energy, contribs
        # Until we figure out the "contrib" version of the code, we must perform a full recompute.
        else:
            energy, contribs =  compute_betre(adj, self.d, self.keep_diag)
            return energy, contribs

# ASquaredD experiences numeric overflow in high dimensions.
# Taking the log of this number reduces growth to be more like N**2 rather than a**n**2.
class LogASquaredD(ASquaredD):
    def __init__(self, d, **kwargs):
        super(LogASquaredD, self).__init__(d, **kwargs)
    def __call__(self, adj, prev_contrib = None, changed_sites = None):
        energy, contrib = super(LogASquaredD, self).__call__(adj, prev_contrib, changed_sites)
        return np.log(energy), contrib
