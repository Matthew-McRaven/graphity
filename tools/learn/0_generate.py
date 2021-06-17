import argparse
import random

import matplotlib.pyplot as plt
import networkx as nx
from networkx.algorithms import clique

import graphity.environment.graph
import graphity.data

def main(args):
	graph_size, clique_size = args.graph_size[0], args.clique_size[0]
	count = args.count[0]
	print(graph_size, clique_size)
	if args.type == "pure": 
		dataset=graphity.data.create_pure_dataset(count, clique_size, graph_size)
		data_dir = f"data/pure/({clique_size}-{graph_size})"
	elif args.type == "impure": 
		dataset=graphity.data.create_impure_dataset(count, clique_size, graph_size)
		data_dir = f"data/impure/({clique_size}-{graph_size})"
	graphity.data.save_dataset(dataset, data_dir)


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Create a graph dataset.')
	parser.add_argument('-c', '--count', required=True, type=int, nargs=1, help='The number of graphs to generate.')
	parser.add_argument('-g', '--graph_size', required=True, type=int, nargs=1, help='The number nodes in each generated graph.')
	parser.add_argument('-k', '--clique_size', required=True, type=int, nargs=1, help='The maximal clique size in each generated graph.')
	group = parser.add_mutually_exclusive_group(required=True)
	group.add_argument("--pure", dest='type', action='store_const', const="pure")
	group.add_argument("--impure", dest='type', action='store_const', const="impure")
	args = parser.parse_args()
	main(args)