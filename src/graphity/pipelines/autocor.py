import functools

import ignite.engine
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss, accuracy
from ignite.utils import setup_logger
import torch

import graphity.environment.lattice

class sync_autocorrelation:
	def __init__(self, eq_lattices, beta, H=graphity.environment.lattice.IsingHamiltonian(), sweeps=100):
		self.beta = beta
		self.count = len(eq_lattices)
		self.eq_lattices = eq_lattices
		self.H = H
		self.sweeps = sweeps
	def run(self,):
		tasks = [graphity.pipelines.create_task(idx, self.beta, self.eq_lattices[0].shape) for idx in range(self.count)]
		trajectories = [self.augment(self.eq_lattices[idx], task) for idx, task in enumerate(tasks)]
		autocorrelation_times = []
		for trajectory in trajectories:
			
			
			first_below_zero = [idx for idx, item in enumerate(trajectory) if self.chi(idx, self.sweeps, trajectory)<0][0]
			if first_below_zero <= 0: first_below_zero = 2 
			norm =  self.chi(0, first_below_zero, trajectory)
			act = functools.reduce(lambda sum, t:sum+self.chi(t, first_below_zero, trajectory)/norm, range(first_below_zero))
			autocorrelation_times.append(torch.round(act).type(torch.LongTensor))
		return max(max(autocorrelation_times), 1)
	def chi(self, t, t_max, trajectory):
		scale = (1/t_max - t)
		term_0, term_1, term_2 = 0,0,0
		for t_prime in range(0, t_max - t):
			term_0 += trajectory[t_prime]['state'].float().mean() *  trajectory[t_prime+t]['state'].float().mean()
			term_1 += trajectory[t_prime]['state'].float().mean()
			term_2 += trajectory[t]['state'].float().mean()
		return scale*term_0 - scale*(term_1 * term_2)

	def augment(self, start_state, task):
		states  = []
		def run_single_timestep(engine, timestep):
			task.sample(task, start_states=[start_state], epoch=engine.state.epoch)
			
			aug_data = {}
			aug_data["state"] =  task.trajectories[0].state_buffer[-1]
			aug_data["energy"] = task.trajectories[0].reward_buffer[-1]
			states.extend([aug_data])

		trainer = ignite.engine.Engine(run_single_timestep)

		trainer.run(range(self.sweeps), max_epochs=1)
		return states