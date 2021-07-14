import io
import itertools
import math
import random

import matplotlib.pyplot as plt
import matplotlib.animation
import networkx as nx

import graphity.utils

# Return a random pair of verticies which are connected via an edge.
def adjacent_nodes(T):
	# Pick a random node in T, call it t_1. Then, pick a random neighbor of t_1, call it t_2.
	t_1 = random.choice(list(T.nodes))
	t_2 = random.choice(list(T.neighbors(t_1)))
	return (t_1, t_2)

# If t_1 and t_2 have the exact same clique set, perform edge contraction on nodes t_1 and t_2 in T.
def contract_edge(T, G_cliques, t_1, t_2):
	# If t_1's vertex set does not match t_2's vertex set, then t_1 != t_2.
	# In this case, we can't safely perform edge contraction between the verticies in T.
	assert sorted(G_cliques[t_1]) == sorted(G_cliques[t_2])

	# Get a list of all verticies which have so far been contracted into t_1 and t_2.
	if type(t_1) == int: t_1_components = [t_1]
	else: t_1_components = [int(i) for i in t_1.split(",")]
	if type(t_2) == int: t_2_components = [t_2]
	else: t_2_components = [int(i) for i in t_2.split(",")]

	# Create a newly named node which is comma seperated values of all previously contracted verticies, in sorted order.
	# Use a set()|set() to eliminate duplice vertex numbers.
	t_new_name = ",".join(str(i) for i in sorted(list(set(t_1_components) | set(t_2_components))))
	T.add_node(t_new_name)

	# Must get neighbors before adding edges, or we will be modifying the state dict as we read it.
	t_new_neighbors = list(T.neighbors(t_1))+list(T.neighbors(t_2))
	for vertex in t_new_neighbors: T.add_edge(t_new_name, vertex)

	# Because we passed the assertion G_cliques[t_1] == G_cliques[t_2], we don't have to do any work to create the new vertex set.
	G_cliques[t_new_name] = G_cliques[t_1]
	
	# Verticies t_1 and t_2 no longer exist -- they have been contracted into t_new. 
	# Therefore, must delete any outstanding references to t_1 and t_2.
	del G_cliques[t_1]
	del G_cliques[t_2]
	T.remove_node(t_1)
	T.remove_node(t_2)

# Pick two random adjacent verticies in T, named t_1 and t_2.
# For each vertex, pick one of its corresponding vertecies in G.
# Attempt to perform vertex contraction between these two nodes.
# Since not all vertex pairs are legal (e.g., an vertex which is in both clique sets can't be contracted with itself),
# must repeatedly pick vertex pairs in G until success is had.
# If vertex contraction were to make t_1 and t_2 have the same clique set, edge contraction must be performed in T.
def contract_vertex(T, G_cliques):
	# Pick two adjacent verticies in T.
	t_1, t_2 = adjacent_nodes(T)

	count = 0
	# Pick a vertex pair in t_1 and t_2 to contract.
	# Use a clever assignment expression to merge iteration check and iteration advancement.
	while count:=count+1 < 1000:
		contract_1, contract_2 = random.choice(G_cliques[t_1]), random.choice(G_cliques[t_2])
		# If a vertex is in both t_1 and t_2, then it has already been contracted, and may not be contracted again.
		# If this is not the case, then we have a valid pair, and we can stop looking.
		if contract_1 not in G_cliques[t_2] and contract_2 not in G_cliques[t_1]: break

	# If two adjacent cliques were picked, they must have some vertex that can be merged.
	# If this is not the case, that would mean every vertex in t_1 is also in t_2.
	# This cannot be true, because edge contraction would have merged t_1 and t_2.
	# Therefore, if we hit this condition, it must be an error in our alogirthm (or **very** bad luck).
	assert count < 1000

	# Create a new vertex with all the connections of t_2 and t_1.
	new_node = sorted(list(frozenset(contract_1) | frozenset(contract_2)))

	# All verticies referencing the old vertex pair before contraction must be updated to point to the contracted vertex.
	for clique in G_cliques.values(): 
		start_len = len(clique)
		if contract_1 in clique: clique.remove(contract_1)
		if contract_2 in clique: clique.remove(contract_2)
		if len(clique) != start_len:clique.append(new_node)
		# Checks that both contract_1, and contract_2 were not members of a single clique.
		# If this were the case, vertex contraction would decrease a maximum clique size by 1,
		# which is an illegal move. Rather than attempt to recover, assert and abort. 
		assert len(clique) == start_len

	# If two verticies in T have become completely merged (i.e., their cliques are identical),
	# must perform edge contraction between the verticies. No need to check that t_2's cliques are a subset of t_1's,
	# since they must have the same number of elements (our maximal cliques must always have the same number of elements
	# by definition of purity). Therefore it is sufficent to check that t_1's cliques are a subset of t_2's.
	if all([clique in G_cliques[t_2] for clique in G_cliques[t_1]]): contract_edge(T, G_cliques, t_1, t_2)

	return T, G_cliques

# Get all unique verticies from a list of cliques.
def enumerate_verticies(cliques):
	verticies = set()
	for clique in cliques.values():
		for vertex in clique: verticies.add(frozenset(vertex))
	return verticies

# Convert a list of vertex number into a comma delimited string.
# As contraction progresses, multiple numbers will be assigned to a single vertex.
def list_to_name(vertex): return ",".join(str(i) for i in vertex)

# Given a list of cliques, construct the associated graph.
def graph_from_cliques(G_cliques):
	G = nx.Graph()
	for clique in G_cliques.values():
		# Get the name of each vertex in the clique. Necessary to properly add edges to G.
		verticies = [list_to_name(vertex) for vertex in clique]
		# The list of all edges between the verticies in the clique.
		comb = list(itertools.combinations(verticies,2))
		G.add_edges_from(comb)
	return G

# Create a pure graph of size n with a maximal clique size of k.
def generate(n, k):
	# Create a seed graph with an arbitrary number of free-floating cliques.
	# There is, as-of-yet not guidance on how to pick this number, other than
	# there needing to be at least as many verticies as your target n.
	T = nx.generators.trees.random_tree(math.ceil(2*(n/k)))
	# Each vertex in T corresponds to k verticies in G. G must start with more than n verticies.
	assert k*len(T.nodes()) >= n
	# Gi
	G_cliques = {node:[[i,] for i in range(k*node, k*(node+1))] for node in T.nodes()}
	while len(enumerate_verticies(G_cliques)) > n: T, G_cliques = contract_vertex(T, G_cliques)

	return graph_from_cliques(G_cliques), T

# Generate a bunch of random graphs and see how long it takes.
if __name__ == "__main__": [generate(7, 3) for _ in range(10000)]