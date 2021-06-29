import io
import itertools
import random
from typing import ItemsView

import matplotlib.pyplot as plt
import matplotlib.animation
import networkx as nx
from networkx.algorithms import clique
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

def _do_add_node(G, cliques, maximal_clique_size, graph_size, base_clique, _io):
    # Add a new node and connect it to all remaining nodes in the clique.
    new_node = len(G)
    G.add_node(new_node)
    G.add_edges_from([(new_node, i) for i in base_clique])
    # Enforce that purity is mantained
    # Create (the only) clique involving the new node.
    new_clique = set(base_clique+[new_node])
    print(f"This adds the cliques: {new_clique}", file=_io)
    #print(f"Adding node {new_node} created the clique {new_clique}", file=_io)

    # Append the newly created clique to the list of cliques.
    cliques.append(new_clique)
    #print(f"This brings the clique list to {cliques}", file=_io)
    #print("Done adding node.\n", file=_io)
    return True, cliques

def _add_node(G, cliques, maximal_clique_size, graph_size, _io, do_anim):
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
    print(f"Adding node {len(G)}", file=_io)
    # Select a random clique to be used as a base for the new clique.
    base_clique = list(random.choice(cliques))
    # Select a random node to remove from the existing clique.
    # This prevents the maximal clique size from growing.
    random.shuffle(base_clique)
    success, cliques = _do_add_node(G, cliques, maximal_clique_size, graph_size, base_clique[:-1], _io)
    return cliques, {"t": f"Adding node: {len(G)-1}"}

def _do_add_edge(G, cliques, maximal_clique_size, graph_size, n_1, n_2, _io):
        c_1 = _collate_max_cliques(n_1, cliques)
        c_2 = _collate_max_cliques(n_2, cliques)
        valid, overlap_nodes = True, set()


        for (i,j) in itertools.product(c_1, c_2):
            if len(i|j) <= maximal_clique_size+1: valid=False
            elif len(i&j) == maximal_clique_size-2: overlap_nodes.add(tuple(i&j))
        if valid and len(overlap_nodes):
            print(f"Adding edge {n_1, n_2}", file=_io)
            G.add_edge(n_1, n_2)
            new_cliques = [{j for j in i}|{n_1,n_2} for i in overlap_nodes]
            cliques.extend(new_cliques)
            print(f"This adds the cliques: {new_cliques}", file=_io)
            print(f"{c_1}, {c_2}", file=_io)
            print(f"This brings the clique list to {cliques}", file=_io)
            print("Done adding edge\n", file=_io)
            return True, cliques
        else: return False, cliques

def _add_edge(G, cliques, maximal_clique_size, graph_size, _io, do_anim):
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
        elif G.has_edge(n_1, n_2): continue
        success, new_cliques = _do_add_edge(G, cliques, maximal_clique_size, graph_size, n_1, n_2, _io)

        if not success: continue
        else: return new_cliques , {'t': f"Adding edge ({n_1}, {n_2})"}

    # If we give up, return the original set of cliques.
    return cliques, {}


def _do_remove_edge(G, cliques, maximal_clique_size, graph_size, n_1, n_2, _io):
    # Get all the max cliques that involve the first and second nodes.
    c_1 = _collate_max_cliques(n_1, cliques)
    c_2 = _collate_max_cliques(n_2, cliques)

    filt_not = lambda c: [i for i in c if not (n_1 in i and n_2 in i)]
    filt_has = lambda c: [i for i in c if (n_1 in i and n_2 in i)]
    # Check that both n_1 and n_2 particiapte in some clique that doesn't involve the other
    filt_1, filt_2 = filt_not(c_1), filt_not(c_2)
    if not (len(filt_1) and len(filt_2)): return False, cliques
    # Check that the edge between n_1 and n_2 participates in at most 1 clique.
    # If the edge participates in more than one clique, then the other clique will be downgraded in size.
    filt_1, filt_2 = filt_has(c_1), filt_has(c_2)
    if len(filt_1) > 1 or len(filt_2) > 1: return False, cliques

    #print(f"Removing clique involving {n_1, n_2}", file=_io)
    def can_remove(_n_1, _n_2):
        # In order for a clique to be "okay" with a removal, each edge must participate in another clique.
        s = {frozenset(i) for i in cliques if (_n_1 in i and _n_2 in i)}
        return len(s) > 1

    # Check that 
    removed = True
    for (_n_1, _n_2) in itertools.combinations(filt_1[0], 2):
        # The selected edge only participates in one clique.
        if sorted([n_1, n_2]) == sorted([_n_1, _n_2]): continue
        # And every other edge participates in other cliques.
        elif not can_remove(_n_1, _n_2): removed = False

    if removed:
        #print(f"Removing edge ({n_1}, {n_2})", file=_io)
        G.remove_edge(n_1, n_2)
        #print(f"This removes cliques {filt_1}", file=_io)
        cliques = [x for x in cliques if (x not in filt_1)]
        #print(f"This brings the clique list to {cliques}", file=_io)
        #print("Done removing edge\n", file=_io)
        return True, cliques
    else: return False, cliques

def _remove_edge(G, cliques, maximal_clique_size, graph_size, _io, do_anim):
    """
    Attempt to remove an edge from a pure graph while maintaining purity.
    
    Will sample up to len(G**2) pairs of nodes, or until the an edge is successfully deleted.

    The algorithm is as follows.
    Sample a pair of nodes, n_1 and n_2.
    If there's no edge between those nodes, begin again.
    Get the list/set of cliques involving n_1 and n_2, call them c_1 and c_2.
    
    For c_1 and c_2, check that if all cliques containg n_1 and n_2 are removed either that c_1 and c_2 have at least one clique left.
    This check is required to prevent a removal for decreasing the maximal clique size of n_1 and n_2.

    For all edges that participate in the selected clique:
    * The selected edge must only be used in one (the selected) clique.
    * All other edges must appear in another clique.
    Failure to meet this condition allows deleting of edges that incorrectly reduce maximal clique size.

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
        if not G.has_edge(n_1, n_2): continue
        success, new_cliques = _do_remove_edge(G, cliques, maximal_clique_size, graph_size, n_1, n_2, _io)

        if not success: continue
        else: return new_cliques , {'t': f"Removing edge ({n_1}, {n_2})"}

    # If we give up, return the original set of cliques.
    return cliques, {}

def random_pure_graph(maximal_clique_size, graph_size, do_anim = True):
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

    _io = io.StringIO()
    # It is impossible to generate a graph with fewer than clique_size number of nodes.
    assert graph_size >= maximal_clique_size

    # Generate a complete graph.
    G = nx.complete_graph(maximal_clique_size)
    # Get a list of all maximal cliques, which is just the set of all nodes.

    cliques = [set(i) for i in nx.find_cliques(G)]
    print(f"Starting with the following cliques: {cliques}", file=_io)

    if do_anim: frames = []
    # Extend the graph until we hit the desired number of nodes.
    while len(G) < graph_size: 
        cliques, items = (random.choice([_add_node, _add_edge, _remove_edge])
            (G, cliques, maximal_clique_size, graph_size, _io, do_anim))
        if not items: continue
        elif do_anim: frames.append({'f':G.copy(), **items})
        #if not graphity.utils.is_pure(G, maximal_clique_size): break

    #if not graphity.utils.is_pure(G, maximal_clique_size): fail()
    #G = random_relabel(G)
    Gt = torch.tensor(nx.to_numpy_array(G))

    def fail():
        fig = plt.figure()
        ax = plt.axes()
        def update(i):
            _G = frames[i]['f']
            ax.clear()
            pos = nx.spring_layout(_G) 
            # Draw nodes & edges.
            nx.draw_networkx_nodes(_G, pos, node_size=700, ax=ax)
            nx.draw_networkx_labels(_G, pos, ax=ax)
            nx.draw_networkx_edges(_G, pos, width=6, ax=ax)
            ax.set_title(frames[i]['t'])

        print(_io.getvalue())
        print(Gt.sum())
        print( Gt)
        print(list(nx.find_cliques(G)))
        graphity.utils.print_as_graph(G)
        print(nx.node_clique_number(G, [i for i in range(len(G))]))
        ani = matplotlib.animation.FuncAnimation(fig, update, interval=1000, frames=len(frames), repeat=False)
        my_writer=matplotlib.animation.PillowWriter(fps=1, codec='libx264', bitrate=2)
        ani.save(filename='gif_test.gif', writer=my_writer)
        assert 0 
    
    #lb, ub = graphity.data.bound_impure(maximal_clique_size, graph_size)
    if not graphity.utils.is_pure(G, maximal_clique_size): fail()
    return Gt

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