import argparse
import os
from pathlib import Path
import random

from deap import base
from deap import creator
from deap import tools

import graphity.data
from graphity.environment.graph import *
class score:
	def __init__(self, eval_dataset, graph_size, net):
		# Must split to perform data augmentation.
		_, eval_dataset = eval_dataset.split([])
		self.eval_dataset = eval_dataset
		self.eval_loader = testloader = torch.utils.data.DataLoader(eval_dataset, batch_size=100)
		self.graph_size = graph_size
		self.net = net
	def __call__(self, ind): 
		self.net.stuff_weights(ind)
		accuracy, _ =  graphity.environment.graph.evaluate(self.net, self.eval_loader, "cpu", count=1000)
		return 100*accuracy,


def evolve_weights(eval_dataset, graph_size, net):

	creator.create("FitnessMax", base.Fitness, weights=(1.0,))
	creator.create("Individual", list, fitness=creator.FitnessMax)

	toolbox = base.Toolbox()

	toolbox.register("attr_int", random.gauss, 0, 3)
	toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_int, n=net.weight_count())
	toolbox.register("population", tools.initRepeat, list, toolbox.individual)

	toolbox.register("mate", tools.cxTwoPoint)
	toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=3, indpb=0.05)
	toolbox.register("select", tools.selTournament, tournsize=3)
	toolbox.register("evaluate", score(eval_dataset, graph_size, net))
	#toolbox.decorate("evaluate", tools.DeltaPenalty(feasible, -2*max_drinks, distance))
	pop = toolbox.population(n=10)
	CXPB, MUTPB, NGEN = 0.5, 0.2, 10

	# Evaluate the entire population
	fitnesses = map(toolbox.evaluate, pop)
	for ind, fit in zip(pop, fitnesses):
		ind.fitness.values = fit

	for g in range(NGEN):
		# Select the next generation individuals
		offspring = toolbox.select(pop, len(pop))
		# Clone the selected individuals
		offspring = list(map(toolbox.clone, offspring))

		# Apply crossover and mutation on the offspring
		for child1, child2 in zip(offspring[::2], offspring[1::2]):
			if random.random() < CXPB:
				toolbox.mate(child1, child2)
				del child1.fitness.values
				del child2.fitness.values

		for mutant in offspring:
			if random.random() < MUTPB:
				toolbox.mutate(mutant)
				del mutant.fitness.values

		# Evaluate the individuals with an invalid fitness
		invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
		fitnesses = map(toolbox.evaluate, invalid_ind)
		for ind, fit in zip(invalid_ind, fitnesses):
			ind.fitness.values = fit

		# The population is entirely replaced by the offspring
		pop[:] = offspring
		print(f"Finished generation {g}.")

	return pop

def main(args):
	
	graph_size, clique_size = args.graph_size, args.clique_size
	pure_dir = f"data/pure/({clique_size}-{graph_size})"
	impure_dir = f"data/impure/({clique_size}-{graph_size})"
	if not os.path.exists(pure_dir) or not os.path.exists(impure_dir): 
		raise ValueError("Dataset directories must exist. Please run 0_generate.py with the same g,k.")
	dataset = graphity.data.FileGraphDataset(pure_dir, impure_dir)


	terms = []
	terms.append(ACoef(rows=4, cols=2))
	terms.append(FACoef(rows=4, cols=2))
	net = SumTerm(terms)

	pop = evolve_weights(dataset, graph_size, net)
	
	best_acc, best_ind = 0,0
	for ind in pop:
		acc = score(dataset, graph_size, net)(ind)[0]
		print(acc, ind)
		if acc > best_acc: 
			best_acc = acc
			best_ind = ind
	net.stuff_weights(best_ind)
	# Save items to disk
	parent = Path("data/models/gp/")
	parent.mkdir(parents=True, exist_ok=True)
	if(os.path.exists(parent/f"({clique_size}-{graph_size}).pth")): os.remove(parent/f"({clique_size}-{graph_size}).pth")
	torch.save(net, parent/f"({clique_size}-{graph_size}).pth")


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Train a NN to recognize graphs')
	parser.add_argument('-g', '--graph_size', required=True, type=int, help='The number nodes in each generated graph.')
	parser.add_argument('-k', '--clique_size', required=True, type=int, help='The maximal clique size in each generated graph.')
	parser.add_argument('--generations', default=10, type=int)
	args = parser.parse_args()
	main(args)
