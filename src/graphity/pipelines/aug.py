import ignite.engine
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss, accuracy
from ignite.utils import setup_logger

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
		data = [self.augment(self.eq_lattics[idx], task) for idx, task in enumerate(tasks)]
		return data
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