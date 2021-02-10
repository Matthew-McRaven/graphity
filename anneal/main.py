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

class TaskWrapper:
	def __init__(self):
		self.epoch = 0
		self.index = 9
		self.task = None

@ray.remote
class controller:
	def __init__(self, task_count):
		self.task_count = 100
		self.available_tasks = [(x, TaskWrapper()) for x in range(self.task_count)]
		for idx, task in self.available_tasks:
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
				number = idx,
				sampler = random_sampler,
				trajectories=1)
			)
			task.task = dist.gather()[0]
		print("Out", flush=True)
		self.mut = Lock()
		self.epoch = 0

	def get_work(self):
		self.mut.acquire()
		rval = None
		try:
			rval = self.available_tasks.pop(0)
		finally:
			self.mut.release()
		return rval
	def cont(self):
		return self.epoch < 200
	def return_work(self, index, task):
		self.mut.acquire()
		if task.epoch > self.epoch:
			self.epoch = task.epoch
		try:
			rval = self.available_tasks.append((index, task))
		finally:
			self.mut.release()

@ray.remote(num_cpus=1)
class worker:
	def __init__(self, ctrl):
		self.ctrl = ctrl

	def train_ground_search(self, task):
		def run_single_timestep(engine, timestep):
			task.sample(task, epoch=engine.state.epoch)
			engine.state.timestep = timestep

		trainer = ignite.engine.Engine(run_single_timestep)

		@trainer.on(Events.EPOCH_STARTED)
		def reset_environment_state(engine):
			task.env.reset()

		@trainer.on(Events.EPOCH_COMPLETED)
		def update_agents(engine):
			pass

		trainer.run(range(5), max_epochs=1)

	def work_cycle(self):
		index, task = ray.get(ctrl.get_work.remote())
		print(f"Working with {index} with epoch {task.epoch}", flush=True)
		self.train_ground_search(task.task)
		task.epoch += 1
		ctrl.return_work.remote(index, task)
		
	def run(self):
		while ray.get(self.ctrl.cont.remote()):
			self.work_cycle()
			
ctrl = controller.remote(100)
# Check that objects can be transferred from each node to each other node.
workers = [worker.remote(ctrl) for _ in range(10)]
ray.get([worker.run.remote() for worker in workers])
