import ray
import numpy as np
import os

import graphity.pipelines
import matplotlib.pyplot as plt

###################################################
# Test than we can perform a ground state search. #
###################################################

def test_sync_eq():
	# Pick parameters that will run quickly but will still yield good looking graphs
	glass_shape = (5,5)
	task_count = 20

	# Unlike anneal/anneal.py, don't distribute computation. It would murder my poor test-runners.

	# Declare storage locations for observables.
	x_axis_data = []
	mags, c, ms = [], [], []

	for beta in np.logspace(-2,.3, 10):
		print(f"beta = {beta}")

		# Create a batch of tasks for equilibriation.
		# This sets up the environment, agent, etc correctly for a given set of parameters.
		# Each individual lattice is a single task.
		tasks = [graphity.pipelines.create_eq_task(idx, beta, glass_shape) for idx in range(task_count)]
		# Simulate all lattices until all are mostly in equilibrium or it becomes apparent that no forward progress is being made.
		# inner_window_size needs to be smaller than outer_window_size/2
		eq_lattices = graphity.pipelines.sync_evolver(tasks, max_epochs=100, inner_window_size=5, outer_window_size=10).run()

		# Compute auto-correlation time
		tau = graphity.pipelines.sync_autocorrelation(eq_lattices, beta, sweeps=10).run()
		print(f"tau={tau}")
		# But clamp it to something small. This is a fast-runnning regression test.
		tau = min(tau, 5)
		# Also clamp the number of samples to something tiny,
		aug_lattices = graphity.pipelines.sync_augmenter(eq_lattices, beta, sweeps=tau*5).run()

		# Record data for chart
		x_axis_data.extend(task_count*[beta])
		mags.extend(graphity.pipelines.magnitization(aug_lattices))
		c.extend(graphity.pipelines.specific_heat(beta, glass_shape)(aug_lattices))
		ms.extend(graphity.pipelines.magnetic_susceptibility(beta, glass_shape)(aug_lattices))
		print(f"beta = {beta}")

	fig, axs = plt.subplots(1,3)

	# Set up magniziation graph.
	# Chose a mostly transparent color for our points.
	# Since we have multiple problem instances per beta, we want more of a density plot of observables rather than
	# a traditional scatter plot
	axs[0].scatter(x_axis_data, mags, alpha=0.1)
	# Data looks terrible when plotted on a linear scale when sampled from a logspace.
	axs[0].set_xscale('log')
	axs[0].set_title('Magniziation vs Beta')
	
	# Set up magnetic susceptibility graph
	axs[1].scatter(x_axis_data, ms, alpha=0.1)
	axs[1].set_xscale('log')
	axs[1].set_title('Magnetic Susceptibility vs Beta')

	# Set up specific heat grpah
	axs[2].scatter(x_axis_data, c, alpha=0.1)
	axs[2].set_xscale('log')
	axs[2].set_title('Specific Heat vs Beta')

	# Create a spot for our ising spin glass picture to live.
	if not os.path.exists("artifacts"):
		os.makedirs("artifacts")
	plt.savefig("artifacts/(5,5)-ising-glass.png")

