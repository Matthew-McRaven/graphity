import itertools
import random

import networkx as nx
import numpy as np
from numpy.random import default_rng
import torch.tensor

import graphity.utils

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

def random_relabel(graph):
    """
    Randomly shuffle the labels of a graph.

    :param graph: A networkx graph that is pure.
    """
    new_labels = [i for i in range(len(graph))]
    random.shuffle(new_labels)
    map_dict = {old:new for (old, new) in enumerate(new_labels)}
    return nx.relabel_nodes(graph, map_dict)


def _collate_max_cliques(node, cliques):
    """
    Get all the cliques that have node as a member

    :param cliques: A list of *all* maximal cliques.
    :param node: A node in a graph whose clique membership is to be determined.
    """
    return [i for i in cliques if node in i]

def _add_node(G, cliques, maximal_clique_size, graph_size):
    """
    Add a node to a pure graph while preserving purity.

        a) Randomly choose a clique from the list of maximal cliques.
        b) Randomly remove one node from the selected clique.
        c) Add a new node to the graph, and add edges between it an all reamining graphs in the selected clique.
           The new node now has the correct maximal clique size (it's connected to k-1 nodes that are all connected to eachother).
           The old nodes maximal clique size is unchanged, since no new connections between existing nodes were added.
        d) Add the new clique to the maximal clique list.
    
    :param G: A networkx graph that is pure.
    :param cliques: A list of *all* maximal cliques.
    :param maximal_clique_size: The max clique size of the pure graph.
    :param graph_size: The total number of nodes in the output graph.
    """
    #print(f"Adding node {len(G)}")
    # Select a random clique to be used as a base for the new clique.
    base_clique = list(random.choice(cliques))
    # Select a random node to remove from the existing clique.
    # This prevents the maximal clique size from growing.
    random.shuffle(base_clique)
    chopped_clique = base_clique[:-1]

    # Add a new node and connect it to all remaining nodes in the clique.
    new_node = len(G)
    G.add_node(new_node)
    G.add_edges_from([(new_node, i) for i in chopped_clique])

    # Create (the only) clique involving the new node.
    new_clique = set(chopped_clique+[new_node])
    #print(new_node, new_clique)

    # Append the newly created clique to the list of cliques.
    cliques.append(new_clique)
    return cliques

def _add_edge(G, cliques, maximal_clique_size, graph_size):
    """
    Attempt to add an edge to a pure graph while maintaining purity.
    
    Will sample up to len(G**2) pairs of nodes, or until the an edge is successfully added.

    The algorithm is as follows.
    Sample a pair of nodes, n_1 and n_2.
    If there's an edge between those nodes, begin again.
    Get the list/set of cliques involving n_1 and n_2, call them c_1 and c_2.
    
    For c_1 and c_2, perform the following check for all pairs: If the intersection of pair_1 and pair_2 has less then 
    clique_size+1 elements, adding an edge would form a clique_size+1 sized clique. Therefore, reject the edge.

    Now we must account for all cliques that will be completed by adding an edge
    First, compute the pairwise union of c_1 and c_2.
    If it the union has two less elements than the max clique size, the "current" set of nodes plus the two selected new nodes will form a clique
    of the appropriate size (why?).

    :param G: A networkx graph that is pure.
    :param cliques: A list of *all* maximal cliques.
    :param maximal_clique_size: The max clique size of the pure graph.
    :param graph_size: The total number of nodes in the output graph.
    """
    attempts = 0
    while attempts < len(G)**2:
        attempts = attempts +1
        n_1, n_2 = random.randint(0, len(G)-1), random.randint(0, len(G)-1)

        if n_1 == n_2: continue
        c_1 = _collate_max_cliques(n_1, cliques)
        c_2 = _collate_max_cliques(n_2, cliques)
        valid, overlap_nodes = True, set()

        if G.has_edge(n_1, n_2): continue 
        for (i,j) in itertools.product(c_1, c_2):
            if len(i|j) <= maximal_clique_size+1: valid=False
            elif len(i&j) == maximal_clique_size-2: overlap_nodes.add(tuple(i&j))
        if valid and len(overlap_nodes):
            #print("Added!!", {first, second})
            G.add_edge(n_1, n_2)
            new_cliques = [{j for j in i}|{n_1,n_2} for i in overlap_nodes]
            #print(new_cliques)
            cliques.extend(new_cliques)
            return cliques
    return cliques

def _remove_edge(G, cliques, maximal_clique_size, graph_size):
    """
    Attempt to remove an edge from a pure graph while maintaining purity.
    
    Will sample up to len(G**2) pairs of nodes, or until the an edge is successfully deleted.

    The algorithm is as follows.
    Sample a pair of nodes, n_1 and n_2.
    If there's no edge between those nodes, begin again.
    Get the list/set of cliques involving n_1 and n_2, call them c_1 and c_2.
    
    For c_1 and c_2, check that if all cliques containg n_1 and n_2 are removed either that c_1 and c_2 have at least one clique left.
    This check is required to prevent a removal for decreasing the maximal clique size of n_1 and n_2.

    Then, compute the c_has_both, which is the set of all cliques from c_1 and c_2 that contain either n_1 or n_2.
    Using this set, construct a set, `check` of all nodes present in c_has_both, removing n_1 and n_2.
    This may check nodes who cannot be influenced by the removal of an edge, but the old algorithm missed some edge (ha) cases.
    For each node in this set, ensure that it has a maximal clique not involving n_1 or n_2.
    This check ensure that a removal of an edge doesn't decrease the max clique size

    :param G: A networkx graph that is pure.
    :param cliques: A list of *all* maximal cliques.
    :param maximal_clique_size: The max clique size of the pure graph.
    :param graph_size: The total number of nodes in the output graph.
    """
    attempts = 0
    while attempts < len(G)**2:
        # Make progress towards termination.
        attempts = attempts +1
        # Sample a pair of nodes
        n_1, n_2 = random.randint(0, len(G)-1), random.randint(0, len(G)-1)
        # Self-loops are disallowed, so there can be no edge added/removed.
        if n_1 == n_2: continue

        # Get all the max cliques that involve the first and second nodes.
        c_1 = _collate_max_cliques(n_1, cliques)
        c_2 = _collate_max_cliques(n_2, cliques)
        valid = True

        if not G.has_edge(n_1, n_2): continue 

        filt = lambda c: [i for i in c if not (n_1 in i and n_2 in i)]
        # Check that both n_1 and n_2 particiapte in some clique that doesn't involve the other
        filt_1, filt_2 = filt(c_1), filt(c_2)
        if not (len(filt_1) and len(filt_2)): return cliques
        # Check every node that interact with n_1 and n_2. This may be expensive,
        # but it is the only way to be sure that we don't mess up some node somewhere.
        # It may be possible to check only the set difference of (c-filt), but I'm not yet sure.
        check = ({x for y in c_1 for x in y} | {x for y in c_2 for x in y}) - {n_1,n_2}
        
        # Check that other nodes have cliques not involving (n_1, n_2).
        for x in check:
            k_x = _collate_max_cliques(x, cliques)
            filtered_x = [i for i in k_x if not (n_1 in i and n_2 in i)]
            valid &= len(filtered_x) > 0
        # Can only remove if all checks pass.
        if valid: 
            G.remove_edge(n_1, n_2)
            return [x for x in cliques if not (n_1 in x and n_2 in x)]
    # If we give up, return the original set of cliques.
    return cliques

def random_pure_graph(maximal_clique_size, graph_size):
    """
    Create a random pure graph.

    The algorithm for growing random pure graphs is as follows:
    1) Start with a complete graph of size `maximal_clique_size`.
    2) Create a list which contains all maximal cliques.
    3) While there are insufficient nodes in the graph either:
        a) Add a node
        b) Attempt to delete an edge. This may fail. If it fails, it will repeat 1/len(G**2), potentially giving
        it the chance to toggle any / every edge.
        c) Attempt to add an edge. Subject to the same failure constraints as b).
    4) Randomly relabel the graph, to prevent the top left corner from being mostly 1's.

    :param maximal_clique_size: The max clique size of the pure graph.
    :param graph_size: The total number of nodes in the output graph.
    """
    # It is impossible to generate a graph with fewer than clique_size number of nodes.
    assert(graph_size >= maximal_clique_size)

    # Generate a complete graph.
    G = nx.complete_graph(maximal_clique_size)
    # Get a list of all maximal cliques, which is just the set of all nodes.

    cliques = [set(i) for i in nx.find_cliques(G)]
    #print(cliques)

    # Extend the graph until we hit the desired number of nodes.
    while len(G) < graph_size: cliques = (random.choice([_add_node, _add_edge, _remove_edge])
        (G, cliques, maximal_clique_size, graph_size))

    #print(cliques)
    G = random_relabel(G)
    return torch.tensor(nx.to_numpy_array(G))

def random_adj_matrix(graph_size, allow_self_loops=False, rng=None, lb=.5, ub=.5):
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
    assert lb >= 0
    assert lb <= ub
    assert ub <= 1
    p = rng.uniform(lb, ub)
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