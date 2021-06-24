import argparse
import itertools
import multiprocessing
import os
from pathlib import Path
import random
import statistics

import matplotlib.pyplot as plt
import networkx as nx
from networkx.algorithms import clique
import torch

import graphity.environment.graph
import graphity.data
from graphity.environment.graph.learn import *

def blargh(args):
	best_acc, best_config = 0,0
	i,j,k,l = args['terms']
	graph_size, clique_size = args['graph_size'], args['clique_size']
	pure_dir, impure_dir = args['pure_dir'], args['impure_dir']
	if not os.path.exists(pure_dir) or not os.path.exists(impure_dir): 
		raise ValueError("Dataset directories must exist. Please run 0_generate.py with the same g,k.")
	def get_nn():
		terms = []
		terms.append(ACoef(rows=i, cols=j))
		terms.append(FACoef(rows=k, cols=l))
		net = SumTerm(terms)
		net = net.to(args['device'])
		return net
	#print(f"With terms {(i,j,k,l)}")
	config, acc, all_acc = graphity.environment.graph.get_best_config(pure_dir, impure_dir, graph_size, clique_size, get_nn,
	batch_size=args['batch_size'], epochs=args['epochs'], dev=args['device'], n_splits=10)
	if best_acc < acc: best_acc, best_config, best_terms = acc, config, (i,j,k,l)
	g, m, s = round(statistics.geometric_mean(all_acc),3),  round(statistics.median(all_acc),3), round(statistics.pstdev(all_acc),3)
	#print(f"{(i,j,k,l)} ::: GMean {g}, Median {m}, STD {s}")
	#print("\n\n")
	return (i,j,k,l), all_acc, best_acc, best_config

def main(args):
	graph_size, clique_size = args.graph_size, args.clique_size

	results = {}
	must_hold = []
	procs, procs_ret = [], []
	best_config, best_terms, best_acc = None, [], 0
	with multiprocessing.Manager() as manager:
		with multiprocessing.Pool(processes=16) as pool:
			for (i,j, k, l) in itertools.product(range(1,5), range(1,5), range(1,5), range(1,5)):
				pargs = manager.dict()
				# MUST NOT DISCARD SOMETHING GIVEN TO US BY MANAGER
				# See: https://stackoverflow.com/a/60795334
				must_hold.append(pargs)
				pargs['terms'] = (i,j,k,l)
				pargs['pure_dir'] = f"data/pure/({clique_size}-{graph_size})"
				pargs['impure_dir'] = f"data/impure/({clique_size}-{graph_size})"
				pargs['clique_size'] = clique_size
				pargs['graph_size'] = graph_size
				pargs['batch_size'] = args.batch_size
				pargs['epochs'] = args.epochs
				pargs['device'] = args.device
				procs.append(pool.apply_async(blargh, (pargs,)))
			for p in procs: procs_ret.append(p.get())
			pool.close()
			pool.join()

	for (terms, all_a, ac, conf) in procs_ret:
		results[terms] = all_a
		if ac > best_acc: best_config, best_terms, best_acc = conf, terms, ac

	for k, v in results.items():
		g, m, s = round(statistics.geometric_mean(v),3),  round(statistics.median(v),3), round(statistics.pstdev(v),3)
		print(f"{k} ::: GMean {g}, Median {m}, STD {s}")
		# Save items to disk
	parent = Path("data/models/nn/")
	parent.mkdir(parents=True, exist_ok=True)
	if(os.path.exists(parent/f"({clique_size}-{graph_size}).pth")): os.remove(parent/f"({clique_size}-{graph_size}).pth")
	torch.save(best_config, parent/f"({clique_size}-{graph_size}).pth")



if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Train a NN to recognize graphs')
	parser.add_argument('-g', '--graph_size', required=True, type=int,  help='The number nodes in each generated graph.')
	parser.add_argument('-k', '--clique_size', required=True, type=int,  help='The maximal clique size in each generated graph.')
	parser.add_argument('--batch-size', default=10, type=int,  help='The number of elements to present the neural net per update.')
	parser.add_argument('--epochs', default=10, type=int,  help='The number of times the entire dataset is visited.')
	parser.add_argument('--device', default="cpu", type=str,  help='The device on which the NN is to be trained.')
	args = parser.parse_args()
	main(args)