import itertools
import functools

import numpy as np
import torch
import torch.distributions

def contribution(adj, d, keep_diag):
    """
    Computes the contribution of each site to the overall energy of a graph.

    :param adj: An adjacency matrix.
    :param d: The value `d` in Tr((A^2-d)^2).
    :param keep_diag: If truthy, mask out the contribution of the diagonal to the energy of the graph.
    """
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
    """
    Computes the Hamiltonian Tr((A^2-d)^2)
    """
    contrib = contribution(adj, d, keep_diag)
    # Sum over contributions, leaving us with a scalar energy.
    return torch.sum(contrib, (0,1)), contrib

class ASquaredD:
    """
    Implement the Hamiltonian Tr((A^2-d)^2).
    Discuess with Betre on 20201015.
    """
    def __init__(self, d, keep_diag=False):
        """
        :param d: The value `d` in Tr((A^2-d)^2).
        :param keep_diag: If truthy, mask out the contribution of the diagonal to the energy of the graph.
        """
        self.d = d
        self.keep_diag = keep_diag

    def contribution(self, adj):
        """
        Determine the contribution of each site to the total energy of the graph.

        :param adj: A tensor containing an undirected graph. 
        """

        return contribution(adj, self.d, self.keep_diag)

    # Signature-compatible with lattice hamiltonians.
    def __call__(self, adj, prev_contribs=None, changed_sites=None):
        """
        Compute the energy of a spacetime graph.

        While lattices have an accelerated computation, we have yet to derive one for graphs.
        Therefore, additional arguments are only kept to remain signature-compatible with spin-glasses

        :param spins: A tensor containing a spin glass. The desired change(s) has already been applied to this glass.
        :param contribs: Ignored.
        :param site: Ignored.
        """
        # At this time, I (MM) don't know how matmul will work in 3+ dims.
        # We will fiure this out when it becomes useful.
        if len(adj.shape) >= 3:
            assert False and "Batched input can have at most 3 dimensions" 

        energy, contribs =  compute_betre(adj, self.d, self.keep_diag)
        return energy, contribs


class LogASquaredD(ASquaredD):
    """
    ASquaredD experiences numeric overflow in high dimensions.
    Taking the log of this number reduces growth to be more like N**2 rather than a**n**2.
    """
    def __init__(self, d, **kwargs):
        super(LogASquaredD, self).__init__(d, **kwargs)
    def __call__(self, adj, prev_contrib = None, changed_sites = None):
        energy, contrib = super(LogASquaredD, self).__call__(adj, prev_contrib, changed_sites)
        return np.log(energy), contrib
