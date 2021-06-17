import argparse
import os
from pathlib import Path
import random

import networkx as nx
from networkx.algorithms import clique
from sklearn.model_selection import KFold
from sklearn import datasets, svm, metrics, decomposition, neighbors, linear_model
from sklearn.model_selection import train_test_split
import torch

import graphity.environment.graph
import graphity.data
def classify(H, dataset):
	clf = svm.SVC()
	data, target = [], []
	for value, label in dataset:
		data.append(H(value.float()).detach().view(-1).numpy() )
		target.append(label)

	X_train, X_test, y_train, y_test = train_test_split(
		data, target, test_size=0.5, shuffle=True)

	clf.fit(X_train, y_train)
	

	predicted = clf.predict(X_test)
	print(f"Classification report for classifier {clf}:\n", f"{metrics.classification_report(y_test, predicted)}\n")
	print(f"Confusion matrix:\n{metrics.confusion_matrix(y_test, clf.predict(X_test))}")

def main(args):
	graph_size, clique_size = args.graph_size[0], args.clique_size[0]

	# Load item from disk
	parent = Path(args.H[0])
	model = torch.load(parent/f"({clique_size}-{graph_size}).pth")
	model.eval()
	

	pure_dir = f"data/pure/({clique_size}-{graph_size})"
	impure_dir = f"data/impure/({clique_size}-{graph_size})"
	if not os.path.exists(pure_dir) or not os.path.exists(impure_dir): 
		raise ValueError("Dataset directories must exist. Please run 0_generate.py with the same g,k.")
	dataset = graphity.data.FileGraphDataset(pure_dir, impure_dir)
	classify(model, dataset)



if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Train a NN to recognize graphs')
	parser.add_argument('-g', '--graph_size', required=True, type=int, nargs=1, help='The number nodes in each generated graph.')
	parser.add_argument('-k', '--clique_size', required=True, type=int, nargs=1, help='The maximal clique size in each generated graph.')
	parser.add_argument('-H', required=True, type=str, nargs=1, help='Base path to the trained Hamiltonian to load.')
	args = parser.parse_args()
	main(args)