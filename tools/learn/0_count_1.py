import argparse
import random

import matplotlib.pyplot as plt
import networkx as nx
from networkx.algorithms import clique

import graphity.environment.graph
import graphity.data

def main(args):
	graph_size, clique_size = args.graph_size, args.clique_size
	print(graph_size, clique_size)

	pure_dir = f"data/pure/({clique_size}-{graph_size})"
	dataset = graphity.data.FileGraphDataset(pure_dir, None)
	_1s = [x[0].mean().item() for x in dataset]
	plt.hist(_1s, bins = 20)
	plt.xlabel("Probability of edge")
	plt.suptitle(f"k={clique_size}, g={graph_size}")
	plt.ylabel("Relative liklihood of each prob.")
	lb, ub = graphity.data.bound_impure(clique_size, graph_size)
	plt.vlines((lb, ub), 0, 100, "red", alpha=.25)
	plt.savefig(f"pure-({clique_size}-{graph_size})-edges")
	plt.show()


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Create a graph dataset.')
	parser.add_argument('-g', '--graph_size', required=True, type=int, help='The number nodes in each generated graph.')
	parser.add_argument('-k', '--clique_size', required=True, type=int, help='The maximal clique size in each generated graph.')
	args = parser.parse_args()
	main(args)