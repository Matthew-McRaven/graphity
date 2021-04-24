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
	ev = graphity.pipelines.sync_evolver(tasks, max_epochs=100, inner_window_size=1, outer_window_size=2,
		run_fn=graphity.pipelines.run_ground, eq_check_fn=None, track_minima=True)
	ev.run()
	print(ev.minima())
	x.extend(task_count*[beta])
	print(f"beta = {beta}")


