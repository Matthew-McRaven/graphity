import random

import networkx as nx
import numpy as np
from numpy.random import default_rng
import torch.tensor


def random_graph(graph_size, rng=None, p=.5):
    """
    Create a random directed graph of a given size.

    :param glass_shape: The required dimensions of the output spin glass.
    :param rng: An optional random number generator. 
    Providing one allows you to get deterministic results out of this function via seeded RNGs.
    Defaults to the numpy's default_rng().
    """
    if rng is None:
        # Need to generate a random nXn graph that is square, symmetric, integer-valued, and all 1's or 0's.
        rng = default_rng()
    # RNG excludes hi endpoint.
    rand = rng.binomial(1, p=p, size=(graph_size, graph_size))

    return torch.tensor(rand)

def random_pure_graph(maximal_clique_size, graph_size):
    """
    Create a random pure graph.

    The algorithm for growing random pure graphs is as follows:
    1) Start with a complete graph of size `maximal_clique_size`.
    2) Create a list which contains all maximal cliques.
    3) While there are insufficient nodes in the graph:
        a) Randomly choose a clique from the list of maximal cliques.
        b) Randomly remove one node from the selected clique.
        c) Add a new node to the graph, and add edges between it an all reamining graphs in the selected clique.
           The new node now has the correct maximal clique size (it's connected to k-1 nodes that are all connected to eachother).
           The old nodes maximal clique size is unchanged, since no new connections between existing nodes were added.
        d) Add the new clique to the maximal clique list.

    :param maximal_clique_size: The max clique size of the pure graph.
    :param graph_size: The total number of nodes in the output graph.
    """
    # It is impossible to generate a graph with fewer than clique_size number of nodes.
    assert(graph_size >= maximal_clique_size)

    # Generate a complete graph.
    G = nx.complete_graph(maximal_clique_size)
    # Get a list of all maximal cliques, which is just the set of all nodes.
    cliques = list(nx.find_cliques(G))

    # Extend the graph until we hit the desired number of nodes.
    for i in range(graph_size-maximal_clique_size):

        # Select a random clique to be used as a base for the new clique.
        base_clique = random.choice(cliques)
        # Select a random node to remove from the existing clique.
        # This prevents the maximal clique size from growing.
        random.shuffle(base_clique)
        chopped_clique = base_clique[:-1]

        # Add a new node and connect it to all remaining nodes in the clique.
        new_node = len(G)+1
        G.add_node(new_node)
        G.add_edges_from([(new_node, i) for i in chopped_clique])

        # Create (the only) clique involving the new node.
        new_clique = chopped_clique+[new_node]

        # Append the newly created clique to the list of cliques.
        cliques.append(new_clique)
    return torch.tensor(nx.to_numpy_array(G))

def random_adj_matrix(graph_size, allow_self_loops=False, rng=None, p=.5):
    """
    Create a random undirected graph.

    :param glass_shape: The required dimensions of the output spin glass.
    :param allow_self_loops: If falsey, diagonal values be forced to 0. Otherwise, they may be 1.
    :param rng: An optional random number generator. 
    Providing one allows you to get deterministic results out of this function via seeded RNGs.
    Defaults to the numpy's default_rng().
    """
    if rng is None:
        # Need to generate a random nXn graph that is square, symmetric, integer-valued, and all 1's or 0's.
        rng = default_rng()
    # RNG excludes hi endpoint.
    rand = random_graph(graph_size, rng, p)
    # Mask out all values on or above the diagonal
    lower_tril = np.tril(rand, -1)
    # TODO: Explain magic of why transpose works.
    upper_tril = lower_tril.transpose()
    # Force the diagonal to be 0 if we disallow self loops.
    diag = np.diag(np.diag(rand)) if allow_self_loops else 0
    # Recombine all tree matrix parts.
    output =  diag + lower_tril + upper_tril
    return torch.tensor(output)