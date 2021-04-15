import pickle


import ray
import numpy as np

import graphity.pipelines

if __name__ == "__main__":
	vals = np.linspace(.1, 3, 40)
	glass_shape = (10,10)
	task_count = 10
	for val in vals:
		beta = val
		print(f"beta = {beta}")
		tasks = [graphity.pipelines.create_task(idx, beta, glass_shape) for idx in range(task_count)]
		eq_graphs = graphity.pipelines.sync_evolver(tasks, max_epochs=10000).run()
		aug_graphs = graphity.pipelines.sync_augmenter(eq_graphs, beta, sweeps=1).run()
		with open(f'beta{round(val, 3)}.pickle', 'wb') as handle:
			pickle.dump(aug_graphs, handle, protocol=pickle.HIGHEST_PROTOCOL)
