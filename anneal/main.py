from collections import Counter
import platform
import time
import ray
import os
from threading import Thread, Lock

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
	glass_shape = (23, 23)
	random_sampler = graphity.task.RandomGlassSampler(glass_shape)
	ss = graphity.strategy.RandomSearch()
	agent = graphity.agent.det.ForwardAgent(ss)	
	dist.add_task(graphity.task.Definition(graphity.task.GraphTask, 
		agent=agent, env=graphity.environment.lattice.SpinGlassSimulator(glass_shape=glass_shape, H=H), 
		episode_length=2*23**2,
		name = "Lingus!!",
		number = index,
		sampler = random_sampler,
		trajectories=1)
	)
	return dist.gather()[0]
	
@ray.remote
class controller:
	def __init__(self, task_count):
		self.task_count = task_count
		self.available_tasks = [create_task(idx) for idx in range(task_count)]
		self.epoch = 0
		self.eq_checks = []
		self.cont = True
		self.resume_state = task_count * [None]

	def run(self):
		while self.cont:
			# Launch eq check and workers before doing join's on either. This helps prevent performance bottlenecks.
			# Check that objects can be transferred from each node to each other node.
			workers = [train_ground_search.remote(
					task.number, self.epoch, self.resume_state[task.number], task
				) for task in self.available_tasks
			]

			# Equilibrium check is expensive and can starve actual work. Don't run too often.
			if self.epoch % 5 == 0: self.eq_checks.append(in_equil.remote(self.epoch, self.available_tasks))
			ready_refs, self.eq_checks = ray.wait(self.eq_checks, num_returns=1, timeout=0.0)
			# Stop iterating if any equilibrium checks passed.
			if any(ray.get(obj) for obj in ready_refs): self.cont = False

			updated_tasks = ray.get(workers)
			self.tasks = [task for task, _ in updated_tasks]

			# Compute where each task should resume on the next epoch.
			for i,_ in enumerate(self.resume_state): self.resume_state[i] = updated_tasks[i][1]['resume']
			
			self.epoch += 1
@ray.remote(num_cpus=1)
def in_equil(epoch, task_list):
	print(f"Checking completion for epoch {epoch}")
	time.sleep(10)
	print(f"Finished checking completion for epoch {epoch}")
	return epoch > 10

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
		print(f"R^bar_({epoch:04d})_{task.number} = {(sum(rewards)/len(rewards)).item():07f}. Best was {min(rewards):03f}.")

	trainer.run(range(1), max_epochs=1)
	ret_state = task.trajectories[0].state_buffer[-1]
	task.clear_trajectories()
	return task, {"resume":ret_state}
		
			
ctrl = controller.remote(10)
ray.get(ctrl.run.remote())

