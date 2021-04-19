import ray
import numpy as np

import graphity.pipelines
import matplotlib.pyplot as plt
if __name__ == "__main__":
	vals = np.logspace(-.9,.3, 10)
	ray.init(address='auto')
	glass_shape = (8,8)
	task_count = 20
	x = []
	mags, c, ms = [], [], []
	for beta in vals:
		print(f"beta = {beta}")
		tasks = [graphity.pipelines.create_task(idx, beta, glass_shape) for idx in range(task_count)]
		eq_lattices = graphity.pipelines.distributed_sync_evolver(tasks, max_epochs=1000, inner_window_size=5, outer_window_size=10).run()
		tau = graphity.pipelines.sync_autocorrelation(eq_lattices, beta, sweeps=100).run()
		print(f"tau={tau}")
		aug_lattices = graphity.pipelines.sync_augmenter(eq_lattices, beta, sweeps=2*tau*20).run()
		x.extend(task_count*[beta])
		mags.extend(graphity.pipelines.magnitization(aug_lattices))
		c.extend(graphity.pipelines.specific_heat(beta, glass_shape)(aug_lattices))
		ms.extend(graphity.pipelines.magnetic_susceptibility(beta, glass_shape)(aug_lattices))
		print(f"beta = {beta}")
	fig, axs = plt.subplots(1,2)
	axs[0].scatter(x, mags, alpha=0.1)
	axs[0].set_xscale('log')
	axs[1].scatter(x, ms, alpha=0.1)
	axs[1].set_xscale('log')
	plt.show()

