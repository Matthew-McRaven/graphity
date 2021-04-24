import ray
import numpy as np
import os

import graphity.pipelines
import matplotlib.pyplot as plt
def test_sync_eq():
	vals = np.logspace(-2,.3, 10)
	glass_shape = (5,5)
	task_count = 20
	x = []
	mags, c, ms = [], [], []
	for beta in vals:
		print(f"beta = {beta}")
		tasks = [graphity.pipelines.create_task(idx, beta, glass_shape) for idx in range(task_count)]
		eq_lattices = graphity.pipelines.sync_evolver(tasks, max_epochs=100, inner_window_size=5, outer_window_size=10).run()
		tau = graphity.pipelines.sync_autocorrelation(eq_lattices, beta, sweeps=10).run()
		print(f"tau={tau}")
		tau = min(tau, 5)
		aug_lattices = graphity.pipelines.sync_augmenter(eq_lattices, beta, sweeps=tau*5).run()
		x.extend(task_count*[beta])
		mags.extend(graphity.pipelines.magnitization(aug_lattices))
		c.extend(graphity.pipelines.specific_heat(beta, glass_shape)(aug_lattices))
		ms.extend(graphity.pipelines.magnetic_susceptibility(beta, glass_shape)(aug_lattices))
		print(f"beta = {beta}")
	fig, axs = plt.subplots(1,2)
	axs[0].scatter(x, mags, alpha=0.1)
	axs[0].set_xscale('log')
	axs[1].scatter(x, c, alpha=0.1)
	axs[1].set_xscale('log')
	if not os.path.exists("artifacts"):
		os.makedirs("artifacts")
	plt.savefig("artifacts/(3,3)-ising-glass.png")

