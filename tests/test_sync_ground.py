import ray
import numpy as np
import os

import graphity.pipelines
import matplotlib.pyplot as plt
def test_sync_eq():
	beta = 1
	glass_shape = (5,5)
	task_count = 2
	x = []
	print(f"beta = {beta}")
	tasks = [graphity.pipelines.create_ground_task(idx, beta, glass_shape) for idx in range(task_count)]
	eq_lattices = graphity.pipelines.sync_evolver(tasks, max_epochs=10, inner_window_size=1, outer_window_size=2,
		run_fn=graphity.pipelines.run_ground, eq_check_fn=None).run()
	aug_lattices = graphity.pipelines.sync_augmenter(eq_lattices, beta, sweeps=1).run()
	x.extend(task_count*[beta])
	print(f"beta = {beta}")


