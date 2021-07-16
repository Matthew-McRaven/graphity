import copy
import io
import itertools
import math
import random
import time

import matplotlib.pyplot as plt
import networkx as nx
import torch

import graphity.utils

# Return all pairs of verticies which are connected via an edge.
def all_adjacent_nodes(T):
	for t_1 in T.nodes():
		for t_2 in T.neighbors(t_1):
			if str(t_1) > str(t_2): continue
			yield (t_1, t_2)

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

def enumerate_clique_pairs(t_1, t_2, G_cliques):
	seen = list()
	for contract_1 in G_cliques[t_1]:
		for contract_2 in G_cliques[t_2]:
			if contract_1 > contract_2: continue
			if contract_1 in G_cliques[t_2] or contract_2 in G_cliques[t_1]: continue
			else: yield (contract_1, contract_2)

# Get all unique verticies from a list of cliques.
def enumerate_verticies(cliques):
	verticies = set()
	for clique in cliques.values():
		for vertex in clique: verticies.add(frozenset(vertex))
	return verticies

def contract_graphs(n, T, G_cliques, count):
	if len(enumerate_verticies(G_cliques)) == n: 
		yield G_cliques, count+1
	for (t_1, t_2) in all_adjacent_nodes(T):
		for (contract_1, contract_2) in enumerate_clique_pairs(t_1, t_2, G_cliques):
			count += 1
			new_G_cliques = copy.deepcopy(G_cliques)
			# Create a new vertex with all the connections of t_2 and t_1.
			new_node = sorted(list(frozenset(contract_1) | frozenset(contract_2)))

			# All verticies referencing the old vertex pair before contraction must be updated to point to the contracted vertex.
			for clique in new_G_cliques.values(): 
				start_len = len(clique)
				if contract_1 in clique: clique.remove(contract_1)
				if contract_2 in clique: clique.remove(contract_2)
				if len(clique) != start_len:clique.append(new_node)
				# Checks that both contract_1, and contract_2 were not members of a single clique.
				# If this were the case, vertex contraction would decrease a maximum clique size by 1,
				# which is an illegal move. Rather than attempt to recover, assert and abort. 
				assert len(clique) == start_len

			# Perform edge contraction if necessary.
			if all([clique in new_G_cliques[t_2] for clique in new_G_cliques[t_1]]):
				new_T = copy.deepcopy(T) 
				contract_edge(new_T, new_G_cliques, t_1, t_2)
			else: new_T = T

			for g, count in contract_graphs(n, new_T, new_G_cliques, count):
				yield g, count

def prufer_sequence(length):
	if length == 2: yield (0, 1)
	for sequence in itertools.permutations(range(length-1),length-2):
		yield sequence

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
	"""# Enumerate all possible relabelings of the graph.
	# Change the contracted verticies names to be a series of integers.
	new_labels = [i for i in range(len(G))]
	
	count = 0
	for _ in itertools.permutations(new_labels, len(G)): 
		#if (count:= count+1) == 20: return
		map_dict = {old:new for (old, new) in zip(G.nodes(), new_labels)}
		yield nx.relabel_nodes(G, map_dict)"""

def generate_all(n, k):
	seen_T, seen_G = [], []
	tg = nx.generators.classic.turan_graph(6,3)
	# Create a seed graph with an arbitrary number of free-floating cliques.
	# There is, as-of-yet not guidance on how to pick this number, other than
	# there needing to be at least as many verticies as your target n.
	for seed in prufer_sequence(math.ceil(2*n)):
		T = nx.from_prufer_sequence(seed)
		gen = (nx.is_isomorphic(T, dedup) for dedup in seen_T)
		if not any(gen): seen_T.append(T)
		else: continue
		assert k*len(T.nodes()) >= n
		G_cliques = {node:[[i,] for i in range(k*node, k*(node+1))] for node in T.nodes()}
		count, last_print = 0, 0
		for g, count in contract_graphs(n, T, G_cliques, count):
			if count > last_print+1000: print(f"Graphs: {(last_print:=count)}")
			G = graph_from_cliques(g)
			#for relabel in all_graph_from_cliques(g):
				#if(count:=count+1)%1000==0:print(count)
				#if not relabel in seen_G: seen_G.append(relabel)
			#if not any((nx.is_isomorphic(G, dedup) for dedup in seen_G)): seen_G.append(G)
			if nx.is_isomorphic(tg, G): 
				nx.draw(G)
				plt.savefig("turan.png")
				assert 0

	return seen_G

lst = generate_all(6,3)
print(len(lst))