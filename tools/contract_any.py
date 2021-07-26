import copy
import io
import itertools
import math
import os
import random
import time

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_agraph import to_agraph
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

def contract_graphs(M,k, T, TG_incidence, count, subsequences, sequence):
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

				break_all = False

				# Can't contract verticies if some clique contains c_1 and c_2.
				for clique in TG_incidence.values():
					if contract_1 in clique and contract_2 in clique: break_all = True
				if break_all: continue

				new_TG_incidence = copy.deepcopy(TG_incidence)
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
def seed_graph(M):
	graphs = []
	for filename in os.listdir(f"data/multi/{M}"):
		if filename.endswith(".gml"):
			G=nx.read_gml(os.path.join(f"data/multi/{M}", filename))
			G=nx.line_graph(G)
			G=nx.convert_node_labels_to_integers(G)
			graphs.append(G)
	for graph in graphs:
		 yield nx.Graph(graph)

def enumerate_pure(M, k, visited=-1):
	seen_T, seen_G = [], []
	count, last_print = 0, 0
	# Create a seed graph with an arbitrary number of free-floating cliques.
	# There is, as-of-yet not guidance on how to pick this number, other than
	# there needing to be at least as many verticies as your target n.
	for idx, T in enumerate(seed_graph(M)):
		gen = (nx.is_isomorphic(T, dedup) for dedup in seen_T)
		if any(gen): continue
		print(T.nodes, T.edges)
		TG_incidence = {node:set(i for i in range(k*node, k*(node+1))) for node in T.nodes()}
		# Must not share subsequences between different seed graphs!!!
		t, s = set(), tuple()
		for g, count in contract_graphs(M, k, T, TG_incidence, count, t, s):
			if g is None: continue
			if count > last_print+10: print(f"Graphs: {(last_print:=count)}, {len(seen_G)}, {len(t)}")
			if visited > 0 and visited < count: return seen_G, seen_T

			G = graph_from_incidence(g)
			gen = (nx.is_isomorphic(G, dedup) for dedup in seen_G)
			if not any(gen): 
				seen_G.append(G)
				seen_T.append(T)
	return (seen_G, seen_T)
		
if __name__ == "__main__":
	def main():
		# Enumerate a limited number of graphs.
		# Internally it will shuffle all posibilities, so that it samples from the graph space randomly(ish).
		lst, lst2 = enumerate_pure(4, 3)
		#lst = enumerate_pure(6,3)
		for idx, (g,t) in enumerate(zip(lst, lst2)):
			A1 = to_agraph(t) 
			A1.layout('dot')
			bytes1 = A1.draw(format="png")

			A1io = io.BytesIO()
			A1io.write(bytes1)
			A1io.seek(0)

			f, ax = plt.subplots(2, 1, figsize=(4,8))
			pos = nx.spring_layout(g)
			nx.draw_networkx_nodes(g, pos, ax=ax[0])
			nx.draw_networkx_edges(g, pos, ax=ax[0])
			nx.draw_networkx_labels(g, pos, ax=ax[0])
			ax[0].set_axis_off()
			ax[1].imshow(mpimg.imread(A1io))
			ax[1].set_axis_off()
			plt.savefig(f"{idx}.png")
			plt.close()
			
	main()
	
