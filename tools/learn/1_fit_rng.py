import argparse
import os
from pathlib import Path
import random

import graphity.data
from graphity.environment.graph import *



def main(args):
	graph_size, clique_size = args.graph_size, args.clique_size
	terms = []
	terms.append(ACoef(rows=4, cols=2))
	terms.append(FACoef(rows=4, cols=2))
	net = SumTerm(terms)

	# Save items to disk
	parent = Path("data/models/rng/")
	parent.mkdir(parents=True, exist_ok=True)
	if(os.path.exists(parent/f"({clique_size}-{graph_size}).pth")): os.remove(parent/f"({clique_size}-{graph_size}).pth")
	torch.save(net, parent/f"({clique_size}-{graph_size}).pth")


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Train a NN to recognize graphs')
	parser.add_argument('-g', '--graph_size', required=True, type=int, help='The number nodes in each generated graph.')
	parser.add_argument('-k', '--clique_size', required=True, type=int, help='The maximal clique size in each generated graph.')
	args = parser.parse_args()
	main(args)
