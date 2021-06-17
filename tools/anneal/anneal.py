# All imports from modules we got from pip. Alphabatized, one per line.
import matplotlib.pyplot as plt
import numpy as np
import ray

# All imports from modules we wrote ourselves. Alphabatized, one per line.
import graphity.pipelines.lattice as pipelines
import graphity.environment.lattice

# If you want to save the results to a file, go to the end and uncomment the line `plt.savefig(...)`.
if __name__ == "__main__":
	##########################
	# Modifiable parameters!!#
	##########################
	# Shape of the lattice
	glass_shape = (12,12)
	# Number of lattices to be evolved at once.
	task_count = 20
	# Pick a lattice hamiltonian that you care about
	#H = graphity.environment.lattice.ConstInfiniteRangeHamiltonian()
	H = graphity.environment.lattice.IsingHamiltonian()

	# Connect to our disributed runtime
	ray.init(address='auto')

	# Declare storage locations for observables.
	x_axis_data = []
	mags, c, ms = [], [], []

	# Iterate over betas that are distibuted logrithmically (More closer to left, fewer to the right).
	for beta in np.logspace(-2,.3, 50):
		print(f"beta = {beta}")

		# Create a batch of tasks for equilibriation.
		# This sets up the environment, agent, etc correctly for a given set of parameters.
		# Each individual lattice is a single task.
		tasks = [pipelines.create_eq_task(idx, beta, glass_shape, H=H) for idx in range(task_count)]
		# Simulate all lattices until all are mostly in equilibrium or it becomes apparent that no forward progress is being made.
		# inner_window_size needs to be smaller than outer_window_size/2
		eq_lattices = pipelines.distributed_sync_evolver(tasks, max_epochs=1000, inner_window_size=5, outer_window_size=10).run()

		# Compute auto-correlation time
		tau = pipelines.distributed_sync_autocorrelation(eq_lattices, beta, H, sweeps=10).run()
		print(f"tau={tau}")
		# But clamp it to something reasonable. We don't have forever.
		tau = min(tau, 20)

		# From the collection of lattices that equilibriated above, evolve them for another tau*50 sweeps and collect all of the intermediate steps.
		aug_lattices = pipelines.distributed_sync_augmenter(eq_lattices, beta, H, sweeps=tau*50).run()

		# Record data for charts
		x_axis_data.extend(task_count*[beta])
		mags.extend(pipelines.lattices.magnitization(aug_lattices))
		c.extend(pipelines.specific_heat(beta, glass_shape)(aug_lattices))
		ms.extend(pipelines.magnetic_susceptibility(beta, glass_shape)(aug_lattices))
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
	#plt.savefig("inf-range-ising-glass.png")
	plt.show()

