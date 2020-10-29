# Returns a real-numbered matrix of 
from torch import tensor

import torch

import graphity.environment.toggle

# Compute the 1'st grads of the graph.
# That is, toggle every edge pair from the starting graph.
# Record Δ energy at for each edge pair [i, j].
def graph_gradient(graph, H):
    local_graph = graph.clone().detach()
    current_energy = H(graph)
    grad = torch.zeros(graph.shape)
    for i in range(0, local_graph.shape[-1]):
        for j in range(i+1, local_graph.shape[-1]):
            graphity.environment.toggle.toggle_edge(i, j, local_graph, False)
            grad[i,j] = grad[j,i] = H(local_graph) - current_energy
            graphity.environment.toggle.toggle_edge(i, j, local_graph, False)
    return grad
