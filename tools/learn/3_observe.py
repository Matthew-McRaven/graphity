import argparse
from pathlib import Path

# All imports from modules we got from pip. Alphabatized, one per line.
import matplotlib.pyplot as plt
import numpy as np
import torch

# All imports from modules we wrote ourselves. Alphabatized, one per line.
import graphity.pipelines.graph as pipelines
import graphity.environment.graph
def main(args):
	graph_size, clique_size = args.graph_size[0], args.clique_size[0]
	##########################
	# Modifiable parameters!!#
	##########################
	# Shape of the lattice
	graph_shape = (graph_size, graph_size)
	# Number of lattices to be evolved at once.
	task_count = 20
	# Pick a lattice hamiltonian that you care about
	# Load item from disk
	parent = Path("data/models/nn/")
	model = torch.load(parent/f"({clique_size}-{graph_size}).pth")
	model.eval()
	H = graphity.environment.graph.LearnedH(model)
	H = graphity.environment.graph.LatticeAdaptor(H)
	#H = graphity.environment.lattice.IsingHamiltonian()

	# Declare storage locations for observables.
	x_axis_data = []
	mags, c, ms = [], [], []

	# Iterate over betas that are distibuted logrithmically (More closer to left, fewer to the right).
	for beta in np.logspace(-2,1, 100):
		print(f"beta = {beta}")

		# Create a batch of tasks for equilibriation.
		# This sets up the environment, agent, etc correctly for a given set of parameters.
		# Each individual lattice is a single task.
		tasks = [pipelines.create_eq_task(idx, beta, graph_shape, H=H) for idx in range(task_count)]
		# Simulate all lattices until all are mostly in equilibrium or it becomes apparent that no forward progress is being made.
		# inner_window_size needs to be smaller than outer_window_size/2
		eq_lattices = pipelines.sync_evolver(tasks, max_epochs=1000, inner_window_size=5, outer_window_size=10).run()

		# Compute auto-correlation time
		tau = pipelines.sync_autocorrelation(eq_lattices, beta, H, sweeps=100).run()
		print(f"tau={tau}")
		# But clamp it to something reasonable. We don't have forever.
		tau = min(tau, 10)

		# From the collection of lattices that equilibriated above, evolve them for another tau*50 sweeps and collect all of the intermediate steps.
		aug_lattices = pipelines.sync_augmenter(eq_lattices, beta, H, sweeps=tau*10).run()

		# Record data for charts
		x_axis_data.extend(task_count*[beta])
		mags.extend(pipelines.magnitization(aug_lattices))
		c.extend(pipelines.specific_heat(beta, graph_shape)(aug_lattices))
		ms.extend(pipelines.magnetic_susceptibility(beta, graph_shape)(aug_lattices))
		print(f"beta = {beta}")

	# Create a plot object with 3 subfigures in a single row.
	fig, axs = plt.subplots(1,3)

	# Set up magniziation graph.
	# Chose a mostly transparent color for our points.
	# Since we have multiple problem instances per beta, we want more of a density plot of observables rather than
	# a traditional scatter plot
	axs[0].scatter(x_axis_data, mags, alpha=0.1)
	# Data looks terrible when plotted on a linear scale when sampled from a logspace.
	axs[0].set_xscale('log')
	axs[0].set_title('Mag.')
	
	# Set up magnetic susceptibility graph
	axs[1].scatter(x_axis_data, ms, alpha=0.1)
	axs[1].set_xscale('log')
	axs[1].set_title('Mag. Susc.')

	# Set up specific heat grpah
	axs[2].scatter(x_axis_data, c, alpha=0.1)
	axs[2].set_xscale('log')
	axs[2].set_title('Spef. Heat')

	#############################################################
	# Uncomment and pick an appropriate filename for the figure #
	#############################################################
	plt.savefig("magic_mike.png")
	#plt.show()

# If you want to save the results to a file, go to the end and uncomment the line `plt.savefig(...)`.
if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Train a NN to recognize graphs')
	parser.add_argument('-g', '--graph_size', required=True, type=int, nargs=1, help='The number nodes in each generated graph.')
	parser.add_argument('-k', '--clique_size', required=True, type=int, nargs=1, help='The maximal clique size in each generated graph.')
	parser.add_argument('-H', required=True, type=str, nargs=1, help='The trained Hamiltonian to load.')
	args = parser.parse_args()
	main(args)