import argparse
import os
from pathlib import Path
import random

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
		if label == 0:to_add.extend([(0, item) for _ in range(diff//count_0)])
	return to_add
def classify(H, dataset):
	clf = svm.SVC()
	data, target = [], []
	for value, label in dataset:
		data.append(H(value.float()).detach().view(-1).numpy())
		target.append(label)

	X_train, X_test, y_train, y_test = train_test_split(
		data, target, test_size=0.25, shuffle=True)

	# Over sample minority class to prevent wild class imbalance.
	for (label, data) in aug(X_train, y_train):
		X_train.append(data)
		y_train.append(label)
	for (label, data) in aug(X_test, y_test):
		X_test.append(data)
		y_test.append(label)

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