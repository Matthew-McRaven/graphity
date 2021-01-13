import itertools
import copy
# Returns a real-numbered matrix of 
import torch

import graphity.environment.toggle

# Compute the 1'st grads of the graph.
# That is, toggle every edge pair from the starting graph.
# Record Δ energy at for each edge pair [i, j].
def graph_gradient(graph, H, allow_self_loop=False):
    local_graph = graph.clone().detach()
    local_graph.requires_grad_(False)
    grad = torch.zeros(graph.shape, dtype=float)
    dim_i, dim_j = graph.shape
    current_energy = H(graph)
        for (i,j) in itertools.product(range(dim_i), range(dim_j)):
            graphity.environment.toggle.toggle_edge(i, j, local_graph, allow_self_loop)
            toggled_energy = H(local_graph)
            grad[i,j] =  toggled_energy - current_energy
            assert i==j or local_graph[i,j] != graph[i,j]
            assert i==j or (local_graph != graph).any()
            graphity.environment.toggle.toggle_edge(i, j, local_graph, allow_self_loop)
    return grad