import functools

import ignite.engine
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss, accuracy
from ignite.utils import setup_logger
import torch
import ray

import graphity.environment.lattice
from .aug import *
def chi(t, t_max, trajectory):
	scale = (1/t_max - t)
	term_0, term_1, term_2 = torch.tensor(0.),torch.tensor(0.),torch.tensor(0.)
	for t_prime in range(0, t_max - t):
		term_0 += trajectory[t_prime]['state'].float().mean() *  trajectory[t_prime+t]['state'].float().mean()
		term_1 += trajectory[t_prime]['state'].float().mean()
		term_2 += trajectory[t]['state'].float().mean()
	ret = scale*term_0 - scale*(term_1 * term_2)
	assert torch.is_tensor(ret)
	return ret
class sync_autocorrelation:
	def __init__(self, eq_lattices, beta, H=graphity.environment.lattice.IsingHamiltonian(), sweeps=100):
		self.beta = beta
		self.count = len(eq_lattices)
		self.eq_lattices = eq_lattices
		self.H = H
		self.sweeps = sweeps
	def run(self,):
		tasks = [graphity.pipelines.create_eq_task(idx, self.beta, self.eq_lattices[0].shape) for idx in range(self.count)]
		trajectories = [augment(self.eq_lattices[idx], task, self.sweeps) for idx, task in enumerate(tasks)]
		autocorrelation_times = []
		for trajectory in trajectories:
			below_zero = [idx for idx, item in enumerate(trajectory) 
				if chi(idx, self.sweeps, trajectory)<0]#/chi(0, self.sweeps, trajectory)<0]
			first_below_zero = below_zero[0] if below_zero else self.sweeps
			if first_below_zero < 1: first_below_zero = torch.tensor([1]) 
			norm =  chi(0, first_below_zero, trajectory)
			assert torch.is_tensor(norm)
			act = functools.reduce(lambda sum, t:sum+chi(t, first_below_zero, trajectory)/norm, 
				range(first_below_zero), torch.tensor(0))
			assert torch.is_tensor(act)
			autocorrelation_times.append(torch.round(act).type(torch.LongTensor))
		return max(max(autocorrelation_times), 1)


class distributed_sync_autocorrelation:
	def __init__(self, eq_lattices, beta, H=graphity.environment.lattice.IsingHamiltonian(), sweeps=100):
		self.beta = beta
		self.count = len(eq_lattices)
		self.eq_lattices = eq_lattices
		self.H = H
		self.sweeps = sweeps
	def run(self,):
		tasks = [graphity.pipelines.create_eq_task(idx, self.beta, self.eq_lattices[0].shape) for idx in range(self.count)]
		trajectories = [distributed_augment.remote(self.eq_lattices[idx], task, self.sweeps) for idx, task in enumerate(tasks)]
		autocorrelation_times = []
		for trajectory in trajectories:
			trajectory = ray.get(trajectory)
			below_zero = [idx for idx, item in enumerate(trajectory) 
				if chi(idx, self.sweeps, trajectory)/chi(0, self.sweeps, trajectory)<0]
			first_below_zero = below_zero[0] if below_zero else self.sweeps-1	
			if first_below_zero < 1: first_below_zero = torch.tensor([1]) 
			norm =  chi(0, first_below_zero, trajectory)
			assert torch.is_tensor(norm)
			act = functools.reduce(lambda sum, t:sum+chi(t, first_below_zero, trajectory)/norm, 
				range(first_below_zero), torch.tensor(0))
			assert torch.is_tensor(act)
			autocorrelation_times.append(torch.round(act).type(torch.LongTensor))
		return max(max(autocorrelation_times), 1)