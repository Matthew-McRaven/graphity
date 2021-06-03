import ignite.engine
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss, accuracy
from ignite.utils import setup_logger
import ray

import graphity.environment.lattice

def augment(start_state, task, sweeps):
	"""
	Given a seed graph, time-evolve it for a number of sweeps, and record the (state, energy) pair at each sweep.

	:param start_state: Seed graph for the simulation.
	:param task: An object which contains an environment, agent, and possesses the ability to sample rollouts. See graphity.task.
	:param sweeps: Number of (state, energy) pairs to collect.
	"""
	states  = []
	def run_single_timestep(engine, timestep):
		task.sample(task)
		task.agent.end_sweep()
		# Only keep the last state / energy, since any intermediary result would come from inside a sweep.
		# We can't extract results from inside a sweep because the size of a sweep depends on the size of the lattice.
		aug_data = {"state":task.trajectories[0].state_buffer[-1],
			"energy": task.trajectories[0].reward_buffer[-1]}	
		states.append(aug_data)
	trainer = ignite.engine.Engine(run_single_timestep)

	# Only called once to initialize simulation to correct state.
	@trainer.on(Events.EPOCH_STARTED)
	def reset_environment_state(engine):
		task.env.reset(start_state)

	# Prefer to run for multiple timesteps rather than multiple epochs.
	# This prevents the simulator from constantly being reset to the start state.
	trainer.run(range(sweeps), max_epochs=1)
	return states

@ray.remote
def distributed_augment(start_state, task, sweeps):
	"""
	Wrap graphity.pipelines.augment so that it can be run in a distributed fashion.

	Chose not to make augment(...) distributed, so that distributed computation is not necessary to do augmentation.
	
	:param start_state: Seed graph for the simulation.
	:param task: An object which contains an environment, agent, and possesses the ability to sample rollouts. See graphity.task.
	:param sweeps: Number of (state, energy) pairs to collect.
	"""
	return augment(start_state, task, sweeps)

class sync_augmenter:
	"""
	Time-evolve a set of seed graphs and collect per-trajectory statistics. Runs on a single thread.
	"""

	def __init__(self, eq_lattices, beta, H=graphity.environment.lattice.IsingHamiltonian(), sweeps=1):
		"""
		:param eq_lattices: List of seed graphs (that are in equilibrium) which are to be time-evolved.
		:param beta: Inverse temperature of the system being simulated.
		:param H: The Hamiltonian under which the system will be simulated., defaults to the Ising model Hamiltonian.
		:param sweeps: Number of (state, energy) pairs to collect.
		"""

		self.beta = beta
		self.count = len(eq_lattices)
		self.eq_lattices = eq_lattices
		self.H = H
		self.sweeps = sweeps

	def run(self):
		"""
		Perform time evolution on eq_lattices, and return the augmented result.

		Draws configuration information from initializer.
		"""
		tasks = [graphity.pipelines.create_eq_task(idx, self.beta, self.eq_lattices[0].shape) for idx in range(self.count)]
		data = [augment(self.eq_lattices[idx], task, self.sweeps) for idx, task in enumerate(tasks)]
		return data

class distributed_sync_augmenter:
	"""
	Time-evolve a set of seed graphs and collect per-trajectory statistics. Runs distributedly on the Ray runtime.
	"""
	def __init__(self, eq_lattices, beta, H=graphity.environment.lattice.IsingHamiltonian(), sweeps=1):
		"""
		:param eq_lattices: List of seed graphs (that are in equilibrium) which are to be time-evolved.
		:param beta: Inverse temperature of the system being simulated.
		:param H: The Hamiltonian under which the system will be simulated., defaults to the Ising model Hamiltonian.
		:param sweeps: Number of (state, energy) pairs to collect.
		"""
		self.beta = beta
		self.count = len(eq_lattices)
		self.eq_lattices = eq_lattices
		self.H = H
		self.sweeps = sweeps

	def run(self,):
		"""
		Perform time evolution on eq_lattices, and return the augmented result.

		Draws configuration information from initializer.
		"""
		tasks = [graphity.pipelines.create_eq_task(idx, self.beta, self.eq_lattices[0].shape) for idx in range(self.count)]
		data = [distributed_augment.remote(self.eq_lattices[idx], task, self.sweeps) for idx, task in enumerate(tasks)]
		# Must convert remotes' data into local copies that can be utilized by our caller.
		data = [ray.get(datum) for datum in data]
		return data