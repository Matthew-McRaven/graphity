import os
from pathlib import Path
import random
import shutil

import torch
from torch.utils.data import Dataset

import graphity.environment.graph.generate
import graphity.utils

class GeneratedGraphDataset(Dataset):
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
	@staticmethod
	def aug_ids(dataset):
		count_0, count_1 = 0,0
		for (label, item) in dataset:
			if label == 0: count_0 += 1
			else: count_1 += 1
		diff = count_1 - count_0
		to_add = []
		if diff // count_0 == 0: return []
		for (label, item) in dataset:
			if label == 0:to_add.extend([(0, item) for _ in range(diff//count_0)])
		return to_add

	def __init__(self, dataset):
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
	def __init__(self, pure_dir, impure_dir):
		things, not_things = [], []
		for f in os.listdir(pure_dir):
			path = (Path(pure_dir)/f).resolve()
			if not path.is_file(): continue
			things.append((0,torch.load(path)))
		for f in os.listdir(impure_dir):
			path = (Path(impure_dir)/f).resolve()
			if not path.is_file(): continue
			not_things.append((1,torch.load(path)))

		self.data = things+not_things
		random.shuffle(self.data)
		self.count = len(things) + len(not_things)
	def split(self, split_ids):
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


def create_pure_dataset(count, clique_size, graph_size):
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
	
def save_dataset(dataset, dir):
	# Save items to disk
	parent = Path(dir)
	# Delete all existing data items, and re-create directory
	if parent.exists(): shutil.rmtree(parent)
	parent.mkdir(parents=True, exist_ok=False)
	for idx, thing in enumerate(dataset): torch.save(thing, parent/f"{idx}.ten")

def create_impure_dataset(count, clique_size, graph_size):
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