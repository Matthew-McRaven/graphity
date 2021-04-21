import ignite.engine
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss, accuracy
from ignite.utils import setup_logger
import ray

def augment(start_state, task, sweeps):
	states  = []
	def run_single_timestep(engine, timestep):
		task.sample(task)
		aug_data = {"state":task.trajectories[0].state_buffer[-1],
			"energy": task.trajectories[0].reward_buffer[-1]}	
		states.append(aug_data)
	trainer = ignite.engine.Engine(run_single_timestep)

	@trainer.on(Events.EPOCH_STARTED)
	def reset_environment_state(engine):
		task.env.reset(start_state)


	trainer.run(range(sweeps), max_epochs=1)
	return states
@ray.remote
def distributed_augment(start_state, task, sweeps):
	return augment(start_state, task, sweeps)

import graphity.environment.lattice
class sync_augmenter:
	def __init__(self, eq_lattics, beta, H=graphity.environment.lattice.IsingHamiltonian(), sweeps=1):
		self.beta = beta
		self.count = len(eq_lattics)
		self.eq_lattics = eq_lattics
		self.H = H
		self.sweeps = sweeps
	def run(self,):
		tasks = [graphity.pipelines.create_task(idx, self.beta, self.eq_lattics[0].shape) for idx in range(self.count)]
		data = [augment(self.eq_lattics[idx], task, self.sweeps) for idx, task in enumerate(tasks)]
		return data

class distributed_sync_augmenter:
	def __init__(self, eq_lattics, beta, H=graphity.environment.lattice.IsingHamiltonian(), sweeps=1):
		self.beta = beta
		self.count = len(eq_lattics)
		self.eq_lattics = eq_lattics
		self.H = H
		self.sweeps = sweeps
	def run(self,):
		tasks = [graphity.pipelines.create_task(idx, self.beta, self.eq_lattics[0].shape) for idx in range(self.count)]
		data = [distributed_augment.remote(self.eq_lattics[idx], task, self.sweeps) for idx, task in enumerate(tasks)]
		data = [ray.get(datum) for datum in data]
		return data