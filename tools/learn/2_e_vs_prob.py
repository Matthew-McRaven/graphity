import argparse
import os
from pathlib import Path
import random

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold, StratifiedShuffleSplit
from sklearn import datasets, svm, metrics, decomposition, neighbors, linear_model
from sklearn.model_selection import train_test_split
import torch

import graphity.environment.graph
import graphity.data

def aug(data, labels):
	count_0, count_1 = 0,0
	for label in labels:
		if label == 0: count_0 += 1
		else: count_1 += 1

	diff = count_1 - count_0
	to_add = []

	if count_0 == 0 or diff // count_0 == 0: return []

	for (label, item) in zip(labels, data):
		if label == 0:to_add.extend([(0, item) for _ in range(diff//count_1)])
	return to_add
def eval(H, dataset, clique_size, graph_size):
	clf = svm.SVC()
	data_p, target_p = [], []
	data_i, target_i = [], []
	for value, label in dataset:
		if label == 0:
			data_p.append(value.float().mean().detach().cpu().numpy())
			target_p.append(H(value).detach().cpu().numpy())
		elif label == 1:
			data_i.append(value.float().mean().detach().cpu().numpy())
			target_i.append(H(value).detach().cpu().numpy())
	plt.scatter(data_p, target_p, alpha=0.01, color="red")
	plt.scatter(data_i, target_i, alpha=0.01, color="blue")
	lb, ub = graphity.data.bound_impure(clique_size, graph_size)
	plt.axvline(lb, color="red", alpha=.25)
	plt.axvline(ub, color="red", alpha=.25)
	plt.xlabel("E|G|")
	plt.suptitle(f"Edge Probability vs Energy")
	plt.ylabel("H(G)")
	plt.savefig(f"e-vs-prob-{clique_size,graph_size}.png")

def main(args):
	graph_size, clique_size = args.graph_size, args.clique_size

	# Load item from disk
	parent = Path(args.H)
	path = parent/f"({clique_size}-{graph_size}).pth"
	model = torch.load(path.absolute()).to("cpu")
	model.eval()
	

	pure_dir = f"data/pure/({clique_size}-{graph_size})"
	impure_dir = f"data/impure/({clique_size}-{graph_size})"
	if not os.path.exists(pure_dir): 
		raise ValueError("Dataset directories must exist. Please run 0_generate.py with the same g,k.")
	dataset = graphity.data.FileGraphDataset(pure_dir, impure_dir)
	eval(model, dataset, clique_size, graph_size)



if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Train a NN to recognize graphs')
	parser.add_argument('-g', '--graph_size', required=True, type=int, help='The number nodes in each generated graph.')
	parser.add_argument('-k', '--clique_size', required=True, type=int, help='The maximal clique size in each generated graph.')
	parser.add_argument('-H', required=True, type=str, help='Base path to the trained Hamiltonian to load.')
	args = parser.parse_args()
	main(args)