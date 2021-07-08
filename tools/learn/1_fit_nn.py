import argparse
import os
from pathlib import Path
import random

import matplotlib.pyplot as plt
import networkx as nx
from networkx.algorithms import clique
import torch

import graphity.environment.graph
import graphity.data
from graphity.environment.graph.learn import *

def main(args):
	def get_nn():
		terms = []
		terms.append(ACoef(rows=4, cols=2))
		terms.append(FACoef(rows=4, cols=2))
		net = SumTerm(terms)
		net = net.to(args.device)
		return net
	graph_size, clique_size = args.graph_size, args.clique_size
	pure_dir = f"data/pure/({clique_size}-{graph_size})"
	impure_dir = f"data/impure/({clique_size}-{graph_size})"
	if not os.path.exists(pure_dir) or not os.path.exists(impure_dir): 
		raise ValueError("Dataset directories must exist. Please run 0_generate.py with the same g,k.")
	config, _1, _2 = graphity.environment.graph.get_best_config(pure_dir, impure_dir, graph_size, clique_size, get_nn,
		batch_size=args.batch_size, epochs=args.epochs, dev=args.device, n_splits=10)
	# Save items to disk
	parent = Path("data/models/nn/")
	parent.mkdir(parents=True, exist_ok=True)
	if(os.path.exists(parent/f"({clique_size}-{graph_size}).pth")): os.remove(parent/f"({clique_size}-{graph_size}).pth")
	torch.save(config, parent/f"({clique_size}-{graph_size}).pth")


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Train a NN to recognize graphs')
	parser.add_argument('-g', '--graph_size', required=True, type=int,  help='The number nodes in each generated graph.')
	parser.add_argument('-k', '--clique_size', required=True, type=int,  help='The maximal clique size in each generated graph.')
	parser.add_argument('--batch-size', default=10, type=int,  help='The number of elements to present the neural net per update.')
	parser.add_argument('--epochs', default=10, type=int,  help='The number of times the entire dataset is visited.')
	parser.add_argument('--device', default='cpu', type=str,  help='Torch device on which training shall occur.')
	args = parser.parse_args()
	main(args)