import functools

import torch
import ray

from .aug import *
def chi(t, t_max, trajectory):
	"""
	Chi is the the time-displace autocorrelation function.

	It is drawn from "Monte Carlo Methods in Statistical Physics", section 3.3.1

	:param t: The current time (lower bound of summation)
	:param t_max: The maximum time step to be considered.
	:param trajectory: An object that stores timesteps in a time-increasing order.
	It is used to determine the magnetization of the spin glass at a point in time.
	"""
	# All summation terms share this coefficient, so pre-compute it.
	scale = (1/t_max - t)
	term_0, term_1, term_2 = torch.tensor(0.),torch.tensor(0.),torch.tensor(0.)
	for t_prime in range(0, t_max - t):
		# There sum 3 terms in (3.21). Let's represent each one as its own line.
		term_0 += trajectory[t_prime]['state'].float().mean() *  trajectory[t_prime+t]['state'].float().mean()
		term_1 += trajectory[t_prime]['state'].float().mean()
		term_2 += trajectory[t]['state'].float().mean()
	# Factor out scale term to save some computation time.
	ret = scale*(term_0 - (term_1 * term_2))
	assert torch.is_tensor(ret)
	return ret
class sync_autocorrelation:
	"""
	Compute the autocorrelation time of a set of lattices. Runs on a single thread.

	As each lattice may have slighlty different χ, we must return the maximum χ.
	It would be formally incorrect to draw statistics from seed graphs that have not been evolved for a χ.
	"""
	def __init__(self, eq_lattices, beta, H, spawn, sweeps=100):
		"""
		:param eq_lattices: List of seed graphs (that are in equilibrium) for which χ is to be computed.
		:param beta: Inverse temperature of the system being simulated.
		:param H: The Hamiltonian under which the system will be simulated.
		:param spawn: A function which will return a new equilibriation task.
		:param sweeps: Number of (state, energy) pairs to collect for each seed graph.
		"""
		self.beta = beta
		self.count = len(eq_lattices)
		self.eq_lattices = eq_lattices
		self.H = H
		self.spawn = spawn
		self.sweeps = sweeps
	def run(self,):
		"""
		Perform χ computation on eq_lattices, and return the the largest seen χ.

		Draws configuration information from initializer.
		"""
		tasks = [self.spawn(idx, self.beta, self.eq_lattices[0].shape, self.H) for idx in range(self.count)]
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
	"""
	Compute the autocorrelation time of a set of lattices. Runs distributedly on the Ray runtime.

	As each lattice may have slighlty different χ, we must return the maximum χ.
	It would be formally incorrect to draw statistics from seed graphs that have not been evolved for a χ.
	"""
	def __init__(self, eq_lattices, beta, H, spawn, sweeps=100):
		"""
		:param eq_lattices: List of seed graphs (that are in equilibrium) for which χ is to be computed.
		:param beta: Inverse temperature of the system being simulated.
		:param H: The Hamiltonian under which the system will be simulated.
		:param spawn: A function which will return a new equilibriation task.
		:param sweeps: Number of (state, energy) pairs to collect for each seed graph.
		"""
		self.beta = beta
		self.count = len(eq_lattices)
		self.eq_lattices = eq_lattices
		self.H = H
		self.spawn = spawn
		self.sweeps = sweeps
	def run(self,):
		"""
		Perform χ computation on eq_lattices, and return the the largest seen χ.

		Draws configuration information from initializer.
		"""
		tasks = [self.spawn(idx, self.beta, self.eq_lattices[0].shape, self.H) for idx in range(self.count)]
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