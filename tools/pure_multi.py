import copy
import io
import itertools
import math
import random
import time

import matplotlib.pyplot as plt
import networkx as nx
from networkx.algorithms import clique
from networkx.drawing.nx_agraph import to_agraph
import torch

import graphity.utils

def expands_max_clique(k, center, max_cliques):

	# Get all the neighbors of the center vertex, including center.
	# If a k+1 clique was formed, center must participate.
	neighbors = set(i for clique in max_cliques for i in clique if center in clique)
	# Create a set of all edges connecting the center node to its neighbors
	edge_list = set(frozenset([center, i]) for i in neighbors if i != center)
	# Find all neighbors reachable within one step of the neighbors.
	for neighbor in neighbors:
		neighbor_neighbors = set(i for clique in max_cliques for i in clique if neighbor in clique) - set([neighbor])
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
def all_node_pairs(seed):
	for t_1 in shuffle(list(seed.nodes)):
		for t_2 in shuffle(list(seed.nodes)):
			if t_1 >= t_2: continue
			yield (t_1, t_2)

# Get all unique verticies from a list of cliques.
def enumerate_verticies(seed_cliques):
	verticies = set()
	for clique in seed_cliques.values():
		for vertex in clique: verticies.add(vertex)
	return verticies

def canonicalize(sequence):
	return tuple(sorted(sequence, key=lambda x: (-x[0], x[1])))

def connect_graph(M, seed, seed_cliques, count, subsequences, sequence):
	if sequence in subsequences: yield None, count
	elif len(seed.edges) == M: 
		subsequences.add(sequence)
		yield seed, count+1
	elif len(seed.edges) > M: yield None, count
	else:
		for (t_1, t_2) in all_node_pairs(seed):
			assert t_1 < t_2
			new_sequence = canonicalize(sequence+((t_1, t_2),))
			if new_sequence in subsequences: continue
			count += 1
			new_seed_cliques = copy.deepcopy(seed_cliques)
			new_seed_cliques.append([t_1, t_2])
			if expands_max_clique(2, t_1, new_seed_cliques): continue
			new_seed = copy.deepcopy(seed)
			new_seed.add_edge(t_1, t_2)
			for graph, count in connect_graph(M, new_seed, new_seed_cliques, count, subsequences, new_sequence): 
				if graph is None: continue
				else: yield graph, count


def enumerate_pure(N, M, visited=-1):
	seen_G, count = [], 0
	seed = nx.empty_graph(N, nx.MultiGraph())
	for G, count in connect_graph(M, seed, [], count, set(), tuple()):
		gen = (nx.is_isomorphic(G, dedup) for dedup in seen_G)
		if not any(gen): seen_G.append(G)
		elif visited>0 and count>visited: break
	return seen_G



if __name__ == "__main__":
	lst = enumerate_pure(10, 5)
	for idx, G in enumerate(lst):
		nx.write_gml(G, f"{idx}.gml")
		if False:
			G = nx.read_gml(f"{idx}.gml")
			A = to_agraph(G) 
			A.layout('dot')
			A.draw(f'{idx}.png')
		

	print(len(lst))
	
