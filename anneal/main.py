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
from numpy.random import Generator, PCG64

import ignite.engine
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss, accuracy
from ignite.utils import setup_logger


import graphity.agent.mdp, graphity.agent.pg, graphity.agent.det
import graphity.grad
import graphity.environment.lattice
import graphity.environment.biqg
import graphity.train
import graphity.task
import graphity.train.ground
import graphity.strategy
import graphity.train
import graphity.grad
# Sample program that demonstrates how to create an agent & environment.
# Then. train this agent for some number of epochs, determined by our hypers.

if __name__ == "__main__":
	ray.init(address='auto')

def create_task(index):
	dist = graphity.task.TaskDistribution()
	H = graphity.environment.lattice.IsingHamiltonian()
	glass_shape = (8, 8)
	random_sampler = graphity.task.RandomGlassSampler(glass_shape)
	ss = graphity.strategy.RandomSearch()
	agent = graphity.agent.mdp.MetropolisAgent(ss, .5)	
	dist.add_task(graphity.task.Definition(graphity.task.GraphTask, 
		agent=agent, env=graphity.environment.lattice.SpinGlassSimulator(glass_shape=glass_shape, H=H), 
		episode_length=2*12**3,
		name = "Lingus!!",
		number = index,
		sampler = random_sampler,
		trajectories=1)
	)
	return dist.gather()[0]
	
@ray.remote
class controller:
	def __init__(self, task_count, finalizers=[]):
		self.task_count = task_count
		self.available_tasks = [create_task(idx) for idx in range(task_count)]
		self.epoch = 0
		self.eq_checks = []
		self.epoch_stop_at = 2000
		self.epoch_additional = 4
		self.resume_state = task_count * [None]
		self.energies = task_count * [[]]
		self.logs = {task:{"trajectories":[]} for task in range(task_count)}
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
			if self.epoch % 5 == 0: self.eq_checks.append(in_equilibrium.remote(self.epoch, self.energies, 10))
			ready_refs, self.eq_checks = ray.wait(self.eq_checks, num_returns=1, timeout=0.0)
			# Stop iterating if any equilibrium checks passed.
			if any(ray.get(obj) for obj in ready_refs): self.epoch_stop_at = min(self.epoch_stop_at, self.epoch)

			updated_tasks = ray.get(workers)
			self.tasks = [task for task, _ in updated_tasks]
			for task in self.tasks: 
				self.logs[task.number]['trajectories'].append(task.trajectories)
				self.energies[task.number].append(task.trajectories[0].reward_buffer[-1])
			# Compute where each task should resume on the next epoch.
			for i,_ in enumerate(self.resume_state): self.resume_state[i] = updated_tasks[i][1]['resume']
			
			self.epoch += 1

		for finalizer in self.finalizers: finalizer(self.epoch_stop_at, self.epoch, self.logs)
	def cont(self):
		return self.epoch < self.epoch_stop_at + self.epoch_additional

@ray.remote(num_cpus=1)
def in_equilibrium(epoch, energy_list, lookback_length, eps=2):
	num_tasks = len(energy_list)
	num_epochs = len(energy_list[0])
	if  num_epochs < lookback_length: return False
	else:
		accumulator = 0
		for i,j in itertools.product(range(num_tasks), range(num_tasks)):
			if j<i: continue
			i, j = energy_list[i][-(i+1):], energy_list[j][-(j+1):]
			accumulator += functools.reduce(lambda sum,item: sum + abs(item[0]-item[1]), zip(i,j),0)
		print(accumulator)
		return accumulator < eps * lookback_length * (.5*num_tasks**2)
		

@ray.remote(num_cpus=1)
def train_ground_search(index, epoch, start_state, task):
	def run_single_timestep(engine, timestep):
		task.sample(task, epoch=engine.state.epoch)
		engine.state.timestep = timestep

	trainer = ignite.engine.Engine(run_single_timestep)

	@trainer.on(Events.EPOCH_STARTED)
	def reset_environment_state(engine):
		task.env.reset(start_state)

	@trainer.on(Events.EPOCH_COMPLETED)
	def update_agents(engine):
		rewards = len(task.trajectories) * [None]
		for idx, traj in enumerate(task.trajectories): rewards[idx] = sum(traj.reward_buffer)
		# TODO: Figure out how to remap rewards in a sane fashion.
		print(f"R^bar_({epoch:04d})_{task.number} = {(sum(rewards)/len(rewards)).item():07f}. Best was {min(traj.reward_buffer):03f}.")

	trainer.run(range(1), max_epochs=1)
	ret_state = task.trajectories[0].state_buffer[-1]
	return task, {"resume":ret_state}
		
def end_computation(eq_epoch, ending_epoch, logs):
	for log in logs:
		log = logs[log]
		print(len(log['trajectories']))
	print("Finalized!!")

def magnitization(eq_epoch, ending_epoch, task_logs):
	summed_mag = 0
	for task in task_logs:
		task = task_logs[task]
		mag = 0
		for buffer in task['trajectories'][-1]:
			print(buffer.state_buffer[-1])
			mag = buffer.state_buffer[-1].float().mean()
			print(mag)

def specific_heat(eq_epoch, ending_epoch, task_logs):
	summed_mag = 0
	for task in task_logs:
		task = task_logs[task]
		energies = []
		for buffers in task['trajectories'][eq_epoch:]:
			for buffer in buffers:
				energies.extend(buffer.reward_buffer[:])
		summed = functools.reduce(lambda sum, item: sum + item, energies,0)
		squared_sums = functools.reduce(lambda sum, item: sum + item**2, energies,0)
		specific_heat = squared_sums/len(energies) - (summed/len(energies))**2
		print(f"C = {specific_heat}")
		
ctrl = controller.remote(10, [end_computation, magnitization, specific_heat])

ray.get(ctrl.run.remote())

