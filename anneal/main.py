from collections import Counter
import platform
import time
import ray
import os
from threading import Thread, Lock
import itertools, functools

import torch
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.transforms import Compose, ToTensor, Normalize
from torchvision.datasets import MNIST
from torch.distributions import Categorical
import numpy as np
from numpy.random import Generator, PCG64

import ignite.engine
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss, accuracy
from ignite.utils import setup_logger


import graphity.agent.det
import graphity.environment.lattice
import graphity.train
import graphity.task
import graphity.train.ground
import graphity.strategy.site
import graphity.train

# Sample program that demonstrates how to create an agent & environment.
# Then. train this agent for some number of epochs, determined by our hypers.

if __name__ == "__main__":
	ray.init(address='auto')

def create_task(index, beta, glass_shape):
	H = graphity.environment.lattice.IsingHamiltonian()
	random_sampler = graphity.task.RandomGlassSampler(glass_shape)
	ss = graphity.strategy.site.RandomSearch()
	agent = graphity.agent.det.ForwardAgent(lambda x,y:(beta,0), ss)	
	return graphity.task.GlassTask(
		agent=agent, env=graphity.environment.lattice.RejectionSimulator(glass_shape=glass_shape, H=H), 
		episode_length=functools.reduce(lambda prod,item: prod*item, glass_shape,2),
		name = "Lingus!!",
		number = index,
		sampler = random_sampler,
		trajectories=1)
	
@ray.remote
class controller:
	def __init__(self, task_count, beta, glass_shape, finalizers=[]):
		self.task_count = task_count
		self.available_tasks = [create_task(idx, beta, glass_shape) for idx in range(task_count)]
		self.epoch = 0
		self.eq_checks = []
		self.forever = 1000
		self.resume_state =  [None for i in range(task_count)]
		self.outer_window_size = 40
		self.inner_window_size = 10
		self.sliding_window = [[] for i in range(task_count)]
		self.energy_list = [[] for i in range(task_count)]
		self.finalizers = finalizers

	def run(self):
		while self.cont():
			# Launch eq check and workers before doing join's on either. This helps prevent performance bottlenecks.
			# Check that objects can be transferred from each node to each other node.
			workers = [train_ground_search.remote(
					task.number, self.epoch, self.resume_state[task.number], task
				) for task in self.available_tasks
			]

			# Equilibrium check is expensive and can starve actual work. Don't run too often.
			if self.epoch % 20 == 0 and len(self.sliding_window[0]) >= self.outer_window_size: 
				self.eq_checks.append(in_equilibrium.remote(self.epoch, self.energy_list, self.inner_window_size))
			ready_refs, self.eq_checks = ray.wait(self.eq_checks, num_returns=1, timeout=0.0)
			# Stop iterating if any equilibrium checks passed.
			if any(ray.get(obj) for obj in ready_refs): self.forever = min(self.forever, self.epoch)

			updated_tasks = ray.get(workers)
			self.tasks = [task for task, _ in updated_tasks]
			for task in self.tasks: 
				self.sliding_window[task.number].append(task.trajectories)
				self.energy_list[task.number].append(task.trajectories[0][-1].reward)
				if len(self.sliding_window[task.number]) > self.outer_window_size:
					self.sliding_window[task.number].pop(0)
					self.energy_list[task.number].pop(0)
			# Compute where each task should resume on the next epoch.
			for i,_ in enumerate(self.resume_state): self.resume_state[i] = updated_tasks[i][1]['resume']
			
			self.epoch += 1

		for finalizer in self.finalizers: finalizer(self.forever, self.epoch, self.sliding_window)
	def cont(self):
		return self.epoch < self.forever + self.outer_window_size+1

def var(batch):
	summed = functools.reduce(lambda sum, item: sum + item, batch, 0)
	squared_sums = functools.reduce(lambda sum, item: sum + item**2, batch,0)
	return squared_sums/len(batch) - (summed/len(batch))**2

@ray.remote(num_cpus=1)
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
	cond = svar_cm < svar_wni
	if cond: print("!!!!!!\nI eq'ed\n!!!!!!")
	return cond
		

@ray.remote(num_cpus=1)
def train_ground_search(index, epoch, start_state, task):
	def run_single_timestep(engine, timestep):
		task.sample(task, start_states = [start_state], epoch=engine.state.epoch)
		engine.state.timestep = timestep

	trainer = ignite.engine.Engine(run_single_timestep)

	@trainer.on(Events.EPOCH_STARTED)
	def reset_environment_state(engine):
		task.env.reset(start_state)

	@trainer.on(Events.EPOCH_COMPLETED)
	def update_agents(engine):
		trajectories = torch.zeros((len(task.trajectories),len(task.trajectories[0])))
		for idx, traj in enumerate(task.trajectories): 	
			trajectories[idx] = torch.tensor([task.trajectories[idx][i_idx].reward for i_idx in range(len(task.trajectories[idx]))])

		trajectories = trajectories.view(-1)
		# TODO: Figure out how to remap rewards in a sane fashion.
		print(f"R^bar_({epoch:04d})_{task.number} = {(sum(trajectories)/len(trajectories)).item():07f}. Best was {min(trajectories):03f}.")

	trainer.run(range(1), max_epochs=1)
	ret_state = task.trajectories[0][-1].state
	return task, {"resume":ret_state}
		
def end_computation(eq_epoch, ending_epoch, sliding_window):
	print("Finalized!!")

def magnitization(eq_epoch, ending_epoch, sliding_window):
	summed_mag = 0
	for idx, task in enumerate(sliding_window):
		mag = 0
		for buffer in sliding_window[idx][-1]:
			state = buffer[-1].state
			#print(state)
			mag = state.float().mean()
			print(mag)

class specific_heat:
	def __init__(self, beta, glass_shape):
		self.beta = beta
		self.glass_shape = glass_shape


		
	def __call__(self, eq_epoch, ending_epoch, sliding_window):
		num_spins = functools.reduce(lambda prod,item: prod *item, self.glass_shape,1)
		for idx, task in enumerate(sliding_window):
			energies, variances = [], []
			for buffers in task:
				for buffer in buffers:
					energies.extend([buffer[idx].reward for idx in range(len(buffer))])	

			batches = [np.random.choice(energies, len(energies)) for i in range(100)]
			
			# Compute per-batch variance
			for batch in batches:
				variance = var(batch)
				variance *= self.beta**2 / num_spins
				variances.append(variance)

			# SQRT(AVG(C^2)-AVG(C)^2)
			# Compute variance of variances, which is specific hea	
			specific_heat = var(energies) * self.beta**2 / num_spins
			error_c = var(variances)**.5
			# See section 3.4 of online book
			print(f"C = {specific_heat} ± {error_c}")
if __name__ == "__main__":
	glass_shape = (20, 20)
	beta = .9
	ctrl = controller.remote(6, beta, glass_shape, [end_computation, magnitization, specific_heat(beta, glass_shape)])

	ray.get(ctrl.run.remote())

