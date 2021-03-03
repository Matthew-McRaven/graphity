import itertools
import copy
# Returns a real-numbered matrix of 
import torch

# Compute the 1'st grads of the graph.
# That is, toggle every edge pair from the starting graph.
# Record Δ energy at for each edge pair [i, j].
def graph_gradient(graph, H, allow_self_loop=False):
    local_graph = graph.clone().detach()
    current_energy = H(graph)
    grad = torch.zeros(local_graph.shape)
    dim_i, dim_j = local_graph.shape
    if H.decomposable and True:
        for (i,j) in itertools.product(range(dim_i), range(dim_j)): 
            contribution = H.contribution(local_graph)
            new_energy = H.fast_toggle(local_graph, contribution, (i,j))
            new_energy = H.normalize(new_energy.sum())
            grad[i, j] =  new_energy - current_energy
    else:
        for (i,j) in itertools.product(range(dim_i), range(dim_j)):
            graphity.environment.toggle.toggle_edge(i, j, local_graph, allow_self_loop)
            toggled_energy = H(local_graph)
            grad[i,j] =  toggled_energy - current_energy
            assert i==j or local_graph[i,j] != graph[i,j]
            assert i==j or (local_graph != graph).any()
            graphity.environment.toggle.toggle_edge(i, j, local_graph, allow_self_loop)
    return grad

def spin_gradient(spins, H, action_count):
    local_spins = spins.clone().detach()
    grad = torch.zeros(*local_spins.shape)
    dims = [range(x) for x in local_spins.shape]
    contrib = H.contribution(local_spins)
    current_energy = H.normalize(contrib.sum())
    for dim in itertools.product(*dims):
        site_val = local_spins[tuple(dim)]
        local_spins[tuple(dim)] *= -1
        new_energy, new_contrib = H(local_spins, prev_contribs=contrib, changed_sites=[(dim, site_val)])
        local_spins[tuple(dim)] *= -1
        grad[tuple(dim)] =  new_energy - current_energy
    return grad
