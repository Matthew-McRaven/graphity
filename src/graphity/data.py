import os
from pathlib import Path
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
	def __init__(self, clique_size, graph_size, count, transform=None, target_transform=None):
		self.count = count
		things = [(0, graphity.environment.graph.random_pure_graph(clique_size, graph_size)) for i in range(count//2)]
		p_edge = things[0][1].float().mean()
		not_things = [(1, graphity.environment.graph.random_adj_matrix(things[0][1].shape[0], p=p_edge)) for i in range(count-len(things))]
		self.data = things+not_things
		random.shuffle(self.data)
		self.transform = transform
		self.target_transform = target_transform

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		label, image = self.data[idx]
		if self.transform:
			image = self.transform(image)
		if self.target_transform:
			label = self.target_transform(label)
		return image.float(), label

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
		count_0, count_1 = 0,0
		for (label, item) in dataset:
			if label == 0: count_0 += 1
			else: count_1 += 1
		diff = count_1 - count_0
		to_add = []
		if count_0 == 0 or diff // count_0 == 0: return []
		for (label, item) in dataset:
			if label == 0:to_add.extend([(0, item) for _ in range(diff//count_0)])
		return to_add

	def __init__(self, dataset):
		"""
		:param dataset: An iterable of (label, graph) pairs.
		"""
		to_add = self.aug_ids(dataset)
		dataset.extend(to_add)
		self.data = dataset
		random.shuffle(self.data)
		self.count = len(self.data)

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		label, image = self.data[idx]
		return image.float(), label

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
			things.append((0,torch.load(path)))
		if impure_dir:
			for f in os.listdir(impure_dir):
				path = (Path(impure_dir)/f).resolve()
				if not path.is_file(): continue
				not_things.append((1,torch.load(path)))

		self.data = things+not_things
		random.shuffle(self.data)
		self.count = len(things) + len(not_things)

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
		label, image = self.data[idx]
		return image.float(), label

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
	for idx, graph in enumerate(dataset):torch.save(graph, parent/f"{idx}.ten")

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
		for dedup in dedup_things:
			if (dedup == thing).all():
				possible = False
				break
		if possible: dedup_things.append(thing)
	return dedup_things

def create_impure_dataset(count, clique_size, graph_size):
	"""
	:param count: The maximum number of grapsh to generate.
	Becuase of de-duplication, the actual number generated may be less.
	:param clique_size: The maximal clique number of the generated graphs.
	:param graph_size: The number of nodes in each graph.
	"""
	# Create a random set of graphs
	# TODO: Sample edge probability from the correct distribution.
	things = [graphity.environment.graph.random_adj_matrix(graph_size, graph_size) for i in range(count)]

	# Remove duplicate
	dedup_things = []
	for thing in things:
		possible = True
		for dedup in dedup_things:
			if not possible: break
			elif (dedup == thing).all(): possible = False
		# Exclude all pure graphs
		if possible and not graphity.utils.is_pure(thing): dedup_things.append(thing)
	return dedup_things