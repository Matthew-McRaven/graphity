import argparse
import multiprocessing
import random

import matplotlib.pyplot as plt
import networkx as nx
from networkx.algorithms import clique

import graphity.environment.graph
import graphity.data
def gen_impure(args):
	count, clique_size, graph_size = args['count'], args['clique_size'], args['graph_size']
	dataset=graphity.data.create_impure_dataset(count, clique_size, graph_size)
	data_dir = f"data/impure/({clique_size}-{graph_size})"
	graphity.data.save_dataset(dataset, data_dir)

def gen_pure(args):
	count, clique_size, graph_size = args['count'], args['clique_size'], args['graph_size']
	dataset=graphity.data.create_pure_dataset(count, clique_size, graph_size)
	data_dir = f"data/pure/({clique_size}-{graph_size})"
	graphity.data.save_dataset(dataset, data_dir)

def main(args):
	must_hold, procs, procs_ret = [], [], []
	clique_size = args.clique_size
	with multiprocessing.Manager() as manager:
		with multiprocessing.Pool(processes=32) as pool:
			for g in range(args.g_min, args.g_max):
				pargs = manager.dict()
				# MUST NOT DISCARD SOMETHING GIVEN TO US BY MANAGER
				# See: https://stackoverflow.com/a/60795334
				must_hold.append(pargs)
				pargs['count'] = args.count
				pargs['clique_size'] = clique_size
				pargs['graph_size'] = g
				procs.append(pool.apply_async(gen_impure, (pargs,)))
				procs.append(pool.apply_async(gen_pure,   (pargs,)))
				#gen_data(pargs)
			# Force processes to terminate.
			for p in procs: procs_ret.append(p.get())
			pool.close()
			pool.join()




if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Create a graph dataset.')
	parser.add_argument('-c', '--count', required=True, type=int, help='The number of graphs to generate.')
	parser.add_argument('--g-min', required=True, type=int, help='The minimum number of graph nodes.')
	parser.add_argument('--g-max', required=True, type=int, help='The maximum number of graph nodes.')
	parser.add_argument('-k', '--clique_size', required=True, type=int, help='The maximal clique size in each generated graph.')
	args = parser.parse_args()
	main(args)