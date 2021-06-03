import functools
import itertools

import ignite.engine
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss, accuracy
from ignite.utils import setup_logger
from numpy.core.fromnumeric import argmin
import numpy as np
from numpy.random import Generator, PCG64
import ray
import torch


from graphity.environment.lattice import reward

from .utils import *
def in_equilibrium(epoch, energy_list, inner_window_size, eps=3):
	"""
	Determine if a set of trajectories are probably in equilibrium.

	#TODO: Come up with an explanation for why this is a sane approach.

	:param epoch: The training/testing epoch are we currently on.
	:param energy_list: A list of each trajectories energies across multiple epochs. That is, a list of lists of energies.
	:param inner_window_size: How many epochs over which to evaluate energies.
	:param eps: If the `sum(variances within each trajectory) < eps`, abort. Modify `eps` to strengthen/weaken the abort condition/ 
	"""
	num_tasks = len(energy_list)
	# A correlation matrix between the (i,j)'th task
	# vm[i,j] is the correlation between task i's ending window and j's starting window.
	cm = torch.zeros((num_tasks, num_tasks))	

	# Compute correlation between starting and ending windows for all pairs.
	# Stands for variance of the i'th ending window.
	var_wni = torch.zeros((num_tasks,))
	for i,j in itertools.product(range(num_tasks), range(num_tasks)):
		# Get the i'th ending window.
		wni = torch.tensor(energy_list[i][-(inner_window_size+1):])
		var_wni[i] = torch.var(wni.view(-1))
		# Get the j'th starting window.
		woj = torch.tensor(energy_list[j][:inner_window_size])
		# Compute the the delta in average energy between the ending and starting windows.
		cm[i,j] = (wni.float().mean()-woj.float().mean())

	# What is the variance of the deltas?
	var_cm = torch.var(cm.view(-1))
	# Rescale the deltas by the number of tasks, so as not to give an unfair eq advantage to a small number of tasks.
	# Computes how different the energies of different tasks are.
	svar_cm = var_cm / num_tasks ** 2
	# Compute the intrinsic variation (due to being a random process) within each trajectory.
	svar_wni = var_wni.mean() / num_tasks
	#print(f"vcm={svar_cm}, vwni={svar_wni}, ad = {abs(svar_cm - svar_wni)}")
	# If there's less variation in all window-pair's energies than there is natural variance from being a random process,
	# we have reach equilibrium. The `svar_wni` term has some unreducable noise, so the only term we can actually affect
	# via simulation / annealing is the variance between trajectories
	cond = svar_cm <= svar_wni
	# If the diagonals are less than some small value, there is no variation within a trajectory.
	# Even if trajectories disagree with each other, we're stuck in a rut and will not make forward progress. ABORT.
	if cond: torch.trace(cm) < eps
	elif abort: print("I aborteded")
	return cond or abort

def run_helper(epoch, start_state, task):
	"""
	Perform one epoch's worth of sweeps to an equilibriation task while recording necessary statistics like energy.

	:param epoch: The training/testing epoch are we currently on.
	:param start_state: Initialize `task`'s environment to this state
	:param task: An equilibriation task.
	"""
	def run_single_timestep(engine, timestep):
		# Defer to the task to sample a trajectory correctly.
		task.sample(task)
		# Let the agent clean up its internal state / modify its annealing schedule.
		task.agent.end_sweep()
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
		#print(f"R^bar_({epoch:04d})_{task.number} = {(sum(trajectories)/len(trajectories)).item():07f}. Best was {min(trajectories):03f}.")
	# Only sample one trajectory, since each timestep is one full rollout / trajectory.
	trainer.run(range(1), max_epochs=1)

def run_eq(epoch, start_state, task):
	"""
	Perform one epoch's worth of sweeps to an equilibriation task while recording necessary statistics like energy.

	Returns the ending state as well as the updated task.

	:param epoch: The training/testing epoch are we currently on.
	:param start_state: Initialize `task`'s environment to this state
	:param task: An equilibriation task.
	"""
	run_helper(epoch, start_state, task)
	ret_state = task.trajectories[0].state_buffer[-1]
	return task, {"resume":ret_state}

def run_ground(epoch, start_state, task):
	"""
	Perform one epoch's worth of sweeps to a ground search task while recording necessary statistics like energy.

	Returns the ending state as well as the updated task.

	:param epoch: The training/testing epoch are we currently on.
	:param start_state: Initialize `task`'s environment to this state
	:param task: An equilibriation task.
	"""
	run_helper(epoch, start_state, task)
	rewards = np.array(task.trajectories[0].reward_buffer[:])
	ret_state = task.trajectories[0].state_buffer[np.argmin(rewards)]
	return task, {"resume":ret_state}

class base_evolver:
	"""
	Provides common functionality for time-evolving and logging a spin glass.
	"""
	def __init__(self, tasks, max_epochs=100, inner_window_size=10, outer_window_size=20, run_fn=run_eq, 
		eq_check_fn=in_equilibrium, track_minima=False):
		"""
		Base initializer for time-evolution
		:param tasks: The set of all tasks which will be time-evolved.
		:param max_epochs: Number of epochs for which tasks will be time evolved.
		:param inner_window_size: The number of epochs over which equilibriation will be computed. 
		The previous window will be compared to the current window for all tasks. This check ensures that the reducible
		variance within a trajectory is minimal
		:param outer_window_size: The number of epochs between equilibriation checks and the distance between current / previous windows.
		:param run_fn: A function which will time-evolve a task by one trajectory's worth of sweeps.
		:param eq_check_fn: A function which takes a list of list of energies as an argument. It returns true if the trajectories
		are in equilibrium, and false otherwise.
		:param track_minima: Should the time-evolution operator track minimal energy states between epochs? Necessary when
		searching for ground states.
		"""
		task_count = len(tasks)
		self.tasks = tasks
		self.epoch = 0
		self.forever = max_epochs
		self.resume_state =  [None for i in range(task_count)]
		self.outer_window_size = outer_window_size
		self.inner_window_size = inner_window_size
		self.sliding_window = [[] for i in range(task_count)]
		self.energy_list = [[] for i in range(task_count)]

		self.run_fn = run_fn
		self.eq_check_fn = eq_check_fn
		self.track_minima = track_minima
		self._minima = []

	def cont(self):
		"""
		Check if our current epoch exceeds the maximum allowable runtime.
		"""
		return self.epoch < self.forever + self.outer_window_size+1

	def log_minima(self, task):
		"""
		Track the minimum energy state for a given task's trajectory.
		"""
		assert self.track_minima
		# TODO: Track all minima if there happens to be more than one.
		mindex = np.argmin(task.trajectories[0].reward_buffer)
		en = task.trajectories[0].reward_buffer[mindex]
		if(mindex+1 < len(task.trajectories[0].reward_buffer)):
			self._minima.append((en, task.trajectories[0].state_buffer[mindex+1]))

	def minima(self):
		"""
		Find all unique minima within the internal buffer of minima.

		Must filter out states which were locally minimal for a trajectory, but not minimal across trajectories.
		"""
		assert self.track_minima
		min_energy = functools.reduce(lambda old, new: min(old, new[0]), self._minima, float("inf"))
		temp = [x for en, x in self._minima if en == min_energy]
		ret = []
		for lattice in temp:
			match = False
			for other in ret: match = match or (other == lattice).all() 
			if match: break
			ret.append(lattice)

		return min_energy, ret

class sync_evolver(base_evolver):
	"""
	Time-evolve tasks using a single-threaded, synchronous computation.

	Tasks are processed in list order.
	"""
	def __init__(self, *args, **kwargs):
		"""
		Let the caller look at base_evolver to figure out what arguments are valid.
		"""
		super(sync_evolver, self).__init__(*args, **kwargs)

	def run(self):
		"""
		Time-evolve tasks (lattices) until they are in equilibrium or a timeout is reached.
		"""
		while self.cont():
			# Perform one epoch's worth of time-evolution.
			updated_tasks = [self.run_fn(self.epoch, self.resume_state[task.number], task)
				for task in self.tasks]	
			# Equilibrium check is expensive and can starve actual work. Don't run too often.
			if (self.epoch % self.inner_window_size == 0 and len(self.sliding_window[0]) >= self.outer_window_size 
				and self.eq_check_fn is not None):
				eq = self.eq_check_fn(self.epoch, self.energy_list, self.inner_window_size)
				if(eq): self.forever = min(self.forever, self.epoch)

			self.tasks = [task for task, _ in updated_tasks]
			# Create a sliding window of energies which lets us track changes in energy within tasks across time.
			for task in self.tasks: 
				self.sliding_window[task.number].append(task.trajectories)
				self.energy_list[task.number].append(task.trajectories[0].reward_buffer[-1])
				# If our sliding window is full, evict the oldest element.
				if len(self.sliding_window[task.number]) > self.outer_window_size:
					self.sliding_window[task.number].pop(0)
					self.energy_list[task.number].pop(0)
				if(self.track_minima): self.log_minima(task)
			# Compute where each task should resume on the next epoch.
			for i,_ in enumerate(self.resume_state): self.resume_state[i] = updated_tasks[i][1]['resume']
			
			self.epoch += 1
		return self.resume_state
class distributed_sync_evolver(base_evolver):
	"""
	Time-evolve tasks using a distributed computation.

	Tasks are scheduled in list order.
	All tasks from one epoch must be completed before any task starts the next epoch.
	"""
	@ray.remote
	def remotify(fn, *args, **kwargs):
		"""
		Hackish helper to turn any function into a distributable / remote function.
		"""
		return fn(*args, **kwargs)

	def __init__(self, *args, **kwargs):
		"""
		Let the caller look at base_evolver to figure out what arguments are valid.
		"""
		super(distributed_sync_evolver, self).__init__(*args, **kwargs)
		self.eq_checks = []

	def run(self):
		"""
		Time-evolve tasks (lattices) until they are in equilibrium or a timeout is reached.
		"""
		while self.cont():
			# Perform one epoch's worth of time-evolution across ray runtime.
			workers = [self.remotify.remote(self.run_fn, self.epoch, self.resume_state[task.number], task) 
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
			# Create a sliding window of energies which lets us track changes in energy within tasks across time.
			for task in self.tasks: 
				self.sliding_window[task.number].append(task.trajectories)
				self.energy_list[task.number].append(task.trajectories[0].reward_buffer[-1])
				# If our sliding window is full, evict the oldest element.
				if len(self.sliding_window[task.number]) > self.outer_window_size:
					self.sliding_window[task.number].pop(0)
					self.energy_list[task.number].pop(0)
				if(self.track_minima): self.log_minima(task)
			# Compute where each task should resume on the next epoch.
			for i,_ in enumerate(self.resume_state): self.resume_state[i] = updated_tasks[i][1]['resume']
			self.epoch += 1
		return self.resume_state