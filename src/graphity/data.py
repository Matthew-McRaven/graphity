from pathlib import Path
import shutil

import torch

import graphity.environment.graph.generate
import graphity.utils

def create_pure_dataset(count, dir, clique_size, graph_size):
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
	
	# Save items to disk
	parent = Path(dir)
	# Delete all existing data items, and re-create directory
	if parent.exists(): shutil.rmtree(parent)
	parent.mkdir(parents=True, exist_ok=False)
	for idx, thing in enumerate(dedup_things): torch.save(thing, parent/f"{idx}.ten")

def create_impure_dataset(count, dir, clique_size, graph_size):
	# Create a random set of graphs
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
	
	# Save items to disk
	parent = Path(dir)
	# Delete all existing data items, and re-create directory
	if parent.exists(): shutil.rmtree(parent)
	parent.mkdir(parents=True, exist_ok=False)
	for idx, thing in enumerate(dedup_things): torch.save(thing, parent/f"{idx}.ten")

def read_pure_dataset(dir): pass