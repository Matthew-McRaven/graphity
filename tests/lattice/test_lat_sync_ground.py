import os

import matplotlib.pyplot as plt
import numpy as np
import ray

import graphity.pipelines
import graphity.environment.lattice

###################################################
# Test than we can perform a ground state search. #
###################################################

def test_sync_eq():
	# Pick parameters that will execute quickly
	beta = 1
	glass_shape = (5,5)
	task_count = 2
	H = graphity.environment.lattice.IsingHamiltonian()
	spawn = graphity.pipelines.lattice.create_ground_task
	to_spin =graphity.pipelines.lattice.to_spin
	# Create a ground state search task and evolve it synchronously.=, non-distributed.
	tasks = [spawn(idx, beta, glass_shape, H) for idx in range(task_count)]
	ev = graphity.pipelines.sync_evolver(tasks, max_epochs=100, inner_window_size=1, outer_window_size=2,
		run_fn=graphity.pipelines.run_ground, eq_check_fn=None, track_minima=True)
	# Run the pipeline to completion and track wahtever minima were found.
	ev.run()
	print(ev.minima())



