import math
import os
from pathlib import Path
import pickle
import random
import shutil

import torch
from torch.utils.data import Dataset

import graphity.environment.graph.generate
import graphity.utils

class GeneratedGraphDataset(Dataset):
	"""
	Make an iterable of pure, impure graphs in memory, and use it a pytorch dataset.
	"""
	def __init__(self, clique_size, graph_size, count):
		self.count = count
		things = [graphity.environment.graph.random_pure_graph(clique_size, graph_size) for i in range(count//2)]
		p_edge = things[0][1].float().mean()
		not_things = [graphity.environment.graph.random_adj_matrix(things[0][1].shape[0], p=p_edge) for i in range(count-len(things))]
		self.data = things+not_things
		random.shuffle(self.data)
		self.bins = set(graph.purity for graph in self.data)

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		graph = self.data[idx]
		return graph.float(), graph.purity

class MemoryGraphDataset(Dataset):
	"""
	Make a random-access iterable of (label, graphs) into a dataset that can be used in pytorch.
	"""
	@staticmethod
	def aug_ids(dataset):
		"""
		Given a binary class, class imbalanced dataset, return a list of items to add the the original dataset
		that would fix the class imbalance.

		The only permitted operation is data replication, since permutation or relabeling may cause elements to be present
		in both training/testing datasets.
		This effect ir most pronounced at small N.
		"""
		bins = {}
		for (purity, item) in dataset:
			if purity not in bins: bins[purity]=1
			else: bins[purity]+=1
		
		total, largest_group = sum(bins.values()), max(bins.values())
		# Determine how many to add from each bin.
		add_sizes = {k:largest_group-v for (k,v) in bins.items()}
		print(add_sizes)
		to_add = []
		for (purity, item) in dataset:
			if add_sizes[purity] == 0: continue
			else: count = largest_group // add_sizes[purity]
			if count == 0: continue
			else: to_add.extend((purity, item) for _ in range(count))
		return to_add, set(bins.keys())

	def __init__(self, dataset):
		"""
		:param dataset: An iterable of (label, graph) pairs.
		"""
		to_add, self.bins = self.aug_ids(dataset)
		dataset.extend(to_add)
		self.data = dataset
		random.shuffle(self.data)
		self.count = len(self.data)

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		purity, graph = self.data[idx]
		return graph.float(), purity

class FileGraphDataset(Dataset):
	"""
	Read a dataset of pure/impure graphs that are stored on the disk as tensors.
	Usable as a torch dataset
	"""
	def __init__(self, pure_dir, impure_dir):
		"""
		:param pure_dir: Directory in which pure graphs (as tensors) are stored.
		No duplicate entries should appear in the dataset.
		However, for computational speed, this condition is not enforced.
		:param impure_dir: Directory in which impure graphs (as tensors) are stored.
		No duplicate or pure graphs should appear in the dataset.
		However, for computational pseed, this condition is not enforced.
		"""
		things, not_things = [], []
		for f in os.listdir(pure_dir):
			path = (Path(pure_dir)/f).resolve()
			if not path.is_file(): continue
			with open(path, "rb") as _f: not_things.append(pickle.load(_f))
		if impure_dir:
			for f in os.listdir(impure_dir):
				path = (Path(impure_dir)/f).resolve()
				if not path.is_file(): continue
				with open(path, "rb") as _f: not_things.append(pickle.load(_f))

		self.data = things+not_things
		random.shuffle(self.data)
		self.count = len(things) + len(not_things)
		self.bins = set(item[0] for item in self.data)

	def split(self, split_ids):
		"""
		Given a list of indecies, split the data into two datasets.

		These datasets will be augmented so that an equal number of both classess will be present in both datasets.
		Augmentation is achieved via replication rather than mutation, since mutation may cause a graph to appear in
		both datasets.

		The first return corresponds to the indecies in split_ids, the second return corresponds to indecies not in split_ids.
		"""
		train_data, test_data = [], []
		for idx, data in enumerate(self.data):
			if idx in split_ids: train_data.append(data)
			else: test_data.append(data)
		return MemoryGraphDataset(train_data), MemoryGraphDataset(test_data)

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		purity, graph = self.data[idx]
		return graph.float(), purity 

def save_dataset(dataset, dir):
	"""
	:param dataset: An above dataset.
	:param dir: A directory in which the dataset is to be serialized.
	"""
	# Save items to disk
	parent = Path(dir)
	# Delete all existing data items, and re-create directory
	if parent.exists(): shutil.rmtree(parent)
	parent.mkdir(parents=True, exist_ok=False)
	for idx, graph in enumerate(dataset):
		with open(parent/f"{idx}.ten", "wb+") as f: pickle.dump((graph.purity, graph), f)

def create_pure_dataset(count, clique_size, graph_size):
	"""
	:param count: The maximum number of grapsh to generate.
	Becuase of de-duplication, the actual number generated may be less.
	:param clique_size: The maximal clique number of the generated graphs.
	:param graph_size: The number of nodes in each graph.
	"""
	# Create a random set of graphs
	things = [graphity.environment.graph.random_pure_graph(clique_size, graph_size) for i in range(count)]
	# Remove duplicate
	dedup_things = []
	for thing in things:
		possible = True
		if not graphity.utils.is_pure(thing, clique_size):
			print(thing, thing.float().mean())
			graphity.utils.print_as_graph(thing)
			assert 0
		for dedup in dedup_things:
			if (dedup == thing).all():
				possible = False
				break
		thing.purity = 1
		if possible: dedup_things.append(thing)
	return dedup_things

def bound_impure(clique_size, graph_size):
	norm = (graph_size * (graph_size - 1))
	nok = math.ceil(graph_size/clique_size)
	lb = 2*clique_size * math.floor(graph_size/clique_size) / norm
	# This is Turán's theorem, which gives the maximum number of edges
	# in a graph of size N using k-partite graphs.
	ub = (clique_size-1)/clique_size * graph_size**2/(norm)
	return lb, ub
	
def create_impure_dataset(count, clique_size, graph_size):
	"""
	:param count: The maximum number of grapsh to generate.
	Becuase of de-duplication, the actual number generated may be less.
	:param clique_size: The maximal clique number of the generated graphs.
	:param graph_size: The number of nodes in each graph.
	"""
	# Create a random set of graphs
	# TODO: Sample edge probability from the correct distribution.
	
	lb, ub = bound_impure(clique_size, graph_size)
	things = [graphity.environment.graph.random_adj_matrix(graph_size, graph_size, lb=lb, ub=ub) for i in range(count)]

	# Remove duplicate
	dedup_things = []
	for thing in things:
		possible = True
		for dedup in dedup_things:
			if not possible: break
			elif (dedup == thing).all(): possible = False
		# Exclude all pure graphs
		thing.purity = purity = graphity.utils.purity_degree(thing, clique_size) 
		if possible and not purity == 1: dedup_things.append(thing)
	return dedup_things