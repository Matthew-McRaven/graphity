import itertools, functools
import ray

import torch
import numpy as np
from numpy.random import Generator, PCG64

import ignite.engine
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss, accuracy
from ignite.utils import setup_logger

from .utils import *
def in_equilibrium(epoch, energy_list, inner_window_size, eps=2):
	num_tasks = len(energy_list)
	cm = torch.zeros((num_tasks, num_tasks))	

	# Compute correlation between starting and ending windows for all pairs.
	var_wni = torch.zeros((num_tasks,))
	for i,j in itertools.product(range(num_tasks), range(num_tasks)):
		wni = torch.tensor(energy_list[i][-(inner_window_size+1):])
		#print(wni)
		var_wni[i] = var(wni.view(-1))
		woj = torch.tensor(energy_list[j][:inner_window_size])
		cm[i,j] = (wni.float().mean()-woj.float().mean())
	var_cm = var(cm.view(-1))
	svar_cm = var_cm / num_tasks ** 2
	svar_wni = var_wni.mean() / num_tasks
	print(f"vcm={svar_cm}, vwni={svar_wni}, ad = {abs(svar_cm - svar_wni)}")
	cond = abs(svar_cm - svar_wni) < .1
	abort_1 = torch.trace(cm) < 3
	abort = abort_1 or False
	if cond: print("!!!!!!\nI eq'ed\n!!!!!!")
	elif abort: print("!!!!!!\nI aborteded\n!!!!!!")
	return cond or abort

def run_eq(index, epoch, start_state, task):
	def run_single_timestep(engine, timestep):
		task.sample(task, start_states = [start_state], epoch=engine.state.epoch)
		engine.state.timestep = timestep

	trainer = ignite.engine.Engine(run_single_timestep)

	@trainer.on(Events.EPOCH_STARTED)
	def reset_environment_state(engine):
		task.env.reset(start_state)

	@trainer.on(Events.EPOCH_COMPLETED)
	def update_agents(engine):
		trajectories = torch.zeros((len(task.trajectories), len(task.trajectories[0])))
		for idx, traj in enumerate(task.trajectories): 	
			trajectories[idx] = torch.tensor(task.trajectories[idx].reward_buffer)
		#print(trajectories)
		trajectories = trajectories.view(-1)
		# TODO: Figure out how to remap rewards in a sane fashion.
		print(f"R^bar_({epoch:04d})_{task.number} = {(sum(trajectories)/len(trajectories)).item():07f}. Best was {min(trajectories):03f}.")

	trainer.run(range(1), max_epochs=1)
	ret_state = task.trajectories[0].state_buffer[-1]
	return task, {"resume":ret_state}

class base_evolver:
	def __init__(self, tasks, run_fn=run_eq, eq_check_fn=in_equilibrium):
		task_count = len(tasks)
		self.tasks = tasks
		self.epoch = 0
		self.forever = 10
		self.resume_state =  [None for i in range(task_count)]
		self.outer_window_size = 4
		self.inner_window_size = 2
		self.sliding_window = [[] for i in range(task_count)]
		self.energy_list = [[] for i in range(task_count)]

		self.run_fn = run_fn
		self.eq_check_fn = eq_check_fn

	def cont(self):
		return self.epoch < self.forever + self.outer_window_size+1

class sync_evolver(base_evolver):
	def __init__(self, tasks):
		super(sync_evolver, self).__init__(tasks)
	def run(self):
		while self.cont():
			# Perform one
			updated_tasks = [self.run_fn( task.number, self.epoch, self.resume_state[task.number], task)
				for task in self.tasks]

			# Equilibrium check is expensive and can starve actual work. Don't run too often.
			if self.epoch % self.inner_window_size == 0 and len(self.sliding_window[0]) >= self.outer_window_size:
				eq = self.eq_check_fn(self.epoch, self.energy_list, self.inner_window_size)
				if(eq): self.forever = min(self.forever, self.epoch)

			self.tasks = [task for task, _ in updated_tasks]
			for task in self.tasks: 
				self.sliding_window[task.number].append(task.trajectories)
				self.energy_list[task.number].append(task.trajectories[0].reward_buffer[-1])
				if len(self.sliding_window[task.number]) > self.outer_window_size:
					self.sliding_window[task.number].pop(0)
					self.energy_list[task.number].pop(0)
			# Compute where each task should resume on the next epoch.
			for i,_ in enumerate(self.resume_state): self.resume_state[i] = updated_tasks[i][1]['resume']
			
			self.epoch += 1
		return self.resume_state

class distributed_sync_evolver(base_evolver):
	@ray.remote
	def remotify(fn, *args, **kwargs):
		return fn(*args, **kwargs)
	def __init__(self, tasks):
		super(distributed_sync_evolver, self).__init__(tasks)
		self.eq_checks = []
	def run(self):
		while self.cont():
			# Launch eq check and workers before doing join's on either. This helps prevent performance bottlenecks.
			# Check that objects can be transferred from each node to each other node.
			workers = [self.remotify.remote(self.run_fn, task.number, self.epoch, self.resume_state[task.number], task) 
				for task in self.tasks
			]

			# Equilibrium check is expensive and can starve actual work. Don't run too often.
			if self.epoch % self.inner_window_size == 0 and len(self.sliding_window[0]) >= self.outer_window_size: 
				self.eq_checks.append(self.remotify.remote(self.eq_check_fn, self.epoch, self.energy_list, 
					self.inner_window_size)
				)
			ready_refs, self.eq_checks = ray.wait(self.eq_checks, num_returns=1, timeout=0.0)
			# Stop iterating if any equilibrium checks passed.
			if any(ray.get(obj) for obj in ready_refs): self.forever = min(self.forever, self.epoch)

			updated_tasks = ray.get(workers)
			self.tasks = [task for task, _ in updated_tasks]
			for task in self.tasks: 
				self.sliding_window[task.number].append(task.trajectories)
				self.energy_list[task.number].append(task.trajectories[0].reward_buffer[-1])
				if len(self.sliding_window[task.number]) > self.outer_window_size:
					self.sliding_window[task.number].pop(0)
					self.energy_list[task.number].pop(0)
			# Compute where each task should resume on the next epoch.
			for i,_ in enumerate(self.resume_state): self.resume_state[i] = updated_tasks[i][1]['resume']
			
			self.epoch += 1
		return self.resume_state