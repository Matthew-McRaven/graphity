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

def eval(H, dataset, clique_size, graph_size):
	d_p, t_p, c_p = [], [], []
	d_i, t_i, c_i = [], [], []
	for value, label in dataset:
		if label == 1.0:
			d_p.append(value.float().mean().detach().cpu().numpy())
			t_p.append(H(value).detach().cpu().numpy())
			c_p.append(graphity.utils.purity_degree(value, clique_size))
		else:
			d_i.append(value.float().mean().detach().cpu().numpy())
			t_i.append(H(value).detach().cpu().numpy())
			c_i.append(graphity.utils.purity_degree(value, clique_size))

	f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, sharex=True)
	ax1.scatter(d_i, t_i, c=c_i, alpha=0.1, cmap='coolwarm', vmin=0, vmax=1)
	ax1.set_title("Impure Graphs")
	ax2.scatter(d_p, t_p, c=c_p, alpha=0.1, cmap='coolwarm', vmin=0, vmax=1)
	ax2.set_title("Pure Graphs")
	ax3.scatter(d_i, t_i, c=c_i, alpha=0.0425, cmap='coolwarm', vmin=0, vmax=1)
	ax3.scatter(d_p, t_p, c=c_p, alpha=0.1, cmap='coolwarm', vmin=0, vmax=1)
	ax3.set_title("All Graphs")
	lb, ub = graphity.data.bound_impure(clique_size, graph_size)
	ax1.axvline(lb, color="red", alpha=.25), ax1.axvline(ub, color="red", alpha=.25)
	ax2.axvline(lb, color="red", alpha=.25), ax2.axvline(ub, color="red", alpha=.25)
	ax3.axvline(lb, color="red", alpha=.25), ax3.axvline(ub, color="red", alpha=.25)
	f.supxlabel("E|G|")
	clb = f.colorbar(plt.cm.ScalarMappable(cmap=plt.get_cmap('coolwarm')), pad=.15)
	clb.ax.set_title("Purity Ratio")
	f.suptitle(f"Edge Probability vs Energy")
	f.supylabel("H(G)")
	f.set_size_inches(8, 6)
	plt.tight_layout()
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