import copy
import io
import itertools
import math
import random
import time

import matplotlib.pyplot as plt
import networkx as nx
from networkx.algorithms import clique
import torch

import graphity.utils

def expands_max_clique(k, center, TG_incidence):

	# Get all the neighbors of the center vertex, including center.
	# If a k+1 clique was formed, center must participate.
	cliques = list(TG_incidence.values())
	neighbors = set(i for clique in cliques for i in clique if center in clique)
	# Create a set of all edges connecting the center node to its neighbors
	edge_list = set(frozenset([center, i]) for i in neighbors if i != center)

	# Find all neighbors reachable within one step of the neighbors.
	for neighbor in neighbors:
		neighbor_neighbors = set(i for clique in cliques for i in clique if neighbor in clique) - set([neighbor])
		edge_list = edge_list| set(frozenset([neighbor, i]) for i in neighbor_neighbors)

	# Enumerate all k+1 size groups of nodes.
	# Check the number of edges that connect only these nodes.
	for neighbor_group in itertools.combinations(neighbors, k+1):
		filtered_list = set(frozenset([i,j]) for (i,j) in edge_list if (i in neighbor_group and j in neighbor_group))
		# Check that there are fewer edges than in a (k+1)-complete graph.
		if len(filtered_list)>=((k+1)*k/2): return True
	return False

def shuffle(lst):
	random.shuffle(lst)
	return lst

# Return all pairs of verticies which are connected via an edge.
def all_adjacent_nodes(T):
	for t_1 in shuffle(list(T.nodes)):
		for t_2 in shuffle(list(T.neighbors(t_1))):
			if t_1 > t_2: continue
			yield (t_1, t_2)

def enumerate_incidence_pairs(t_1, t_2, TG_indicence):
	seen = list()
	for contract_1 in shuffle(list(TG_indicence[t_1])):
		for contract_2 in shuffle(list(TG_indicence[t_2])):
			if contract_1 > contract_2: continue
			if contract_1 in TG_indicence[t_2] or contract_2 in TG_indicence[t_1]: continue
			else: yield (contract_1, contract_2)

# Get all unique verticies from a list of cliques.
def enumerate_verticies(incidence):
	verticies = set()
	for clique in incidence.values():
		for vertex in clique: verticies.add(vertex)
	return verticies

def canonicalize(sequence):
	return tuple(sorted(sequence, key=lambda x: (-x[0], x[1])))

def contract_graphs(M,k, T, TG_incidence, count, subsequences=set(), sequence=tuple()):
	if sequence in subsequences: yield None, count
	else:
		# Don't allow graphs with fewer than M distinct cliques.
		for (clique1, clique2) in itertools.combinations(TG_incidence.values(), 2):
			if all(i in clique2 for i in clique1): break
		else:
			subsequences.add(sequence)
			yield TG_incidence, count+1

		for (t_1, t_2) in all_adjacent_nodes(T):
			for (contract_1, contract_2) in enumerate_incidence_pairs(t_1, t_2, TG_incidence):
				assert contract_1 < contract_2
				count += 1
				new_sequence = canonicalize(sequence+((contract_1, contract_2),))
				if new_sequence in subsequences: continue
				new_TG_incidence = copy.deepcopy(TG_incidence)
				break_all = False

				# Can't contract verticies if some clique contains c_1 and c_2.
				for clique in new_TG_incidence.values():
					if contract_1 in clique and contract_2 in clique: break_all = True
				if break_all: continue

				# All verticies referencing the old vertex pair before contraction must be updated to point to the contracted vertex.
				for clique in new_TG_incidence.values():
					start_len = len(clique)
					if contract_2 in clique: clique.remove(contract_2)
					if len(clique) != start_len: clique.add(contract_1)

				if expands_max_clique(k, contract_1, new_TG_incidence): continue

				# Do not perform edge contraction, since it can reduce max clique size.
				if all([clique in new_TG_incidence[t_2] for clique in new_TG_incidence[t_1]]): yield None, count
				else:
					for g, count in contract_graphs(M, k, T, new_TG_incidence, count, subsequences, new_sequence):
						if g is None: continue
						else: yield g, count

# Given a list of cliques, construct the associated graph.
def graph_from_incidence(TG_incidence):
	G = nx.Graph()
	for clique in TG_incidence.values():
		# Get the name of each vertex in the clique. Necessary to properly add edges to G.
		verticies = [vertex for vertex in clique]
		# The list of all edges between the verticies in the clique.
		comb = list(itertools.combinations(verticies,2))
		G.add_edges_from(comb)
	return G

# Load the seed undirected graphs.
# Cliques live on the edge(s).
def seed_graph(M, connected=False):
	
	if connected: suffix = "c"
	else: suffix = "d1"

	for idx, seed in enumerate(nx.read_graph6(f"data/undirected/{M}{suffix}.g6")):
		yield nx.convert_node_labels_to_integers(nx.line_graph(seed))

def enumerate_pure(M, k, visited=-1):
	seen_T, seen_G = [], []
	count, last_print = 0, 0
	# Create a seed graph with an arbitrary number of free-floating cliques.
	# There is, as-of-yet not guidance on how to pick this number, other than
	# there needing to be at least as many verticies as your target n.
	for idx, T in enumerate(seed_graph(M)):
		TG_incidence = {node:set(i for i in range(k*node, k*(node+1))) for node in T.nodes()}
		# Must not share subsequences between different seed graphs!!!
		for g, count in contract_graphs(M, k, T, TG_incidence, count):
			if g is None: continue
			if count > last_print+1000: print(f"Graphs: {(last_print:=count)}, {len(seen_G)}")
			if visited > 0 and visited < count: return seen_G

			G = graph_from_incidence(g)
			gen = (nx.is_isomorphic(G, dedup) for dedup in seen_G)
			if not any(gen): seen_G.append(G)
	return seen_G
		
if __name__ == "__main__":
	# Enumerate a limited number of graphs.
	# Internally it will shuffle all posibilities, so that it samples from the graph space randomly(ish).
	lst = enumerate_pure(4, 3)
	#lst = enumerate_pure(6,3)
	print(len(lst))
	for idx, g in enumerate(lst):
		pos = nx.spring_layout(g)
		nx.draw_networkx_nodes(g, pos)
		nx.draw_networkx_edges(g, pos)
		nx.draw_networkx_labels(g, pos)
		plt.savefig(f"{idx}.png")
		plt.clf()
	# However, it can also enumerate all possible graphs if not given a count.
	
