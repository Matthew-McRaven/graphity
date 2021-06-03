import functools

import graphity.agent.det
import graphity.environment.lattice
from graphity.strategy.anneal import ConstBeta
import graphity.strategy.site
import graphity.task

############################################
# Helpers to create lattice / glass tasks. #
############################################

def create_eq_task(index, beta, glass_shape, name="EQ",
	H = graphity.environment.lattice.IsingHamiltonian()):
	"""
	Create a task which will evolve random lattices via a random walk and metropolis-hastings acceptance.
	"""
	random_sampler = graphity.task.RandomGlassSampler(glass_shape)
	# Equilibritation must preserve detailed-balance, which means we need to do a random walk.
	ss = graphity.strategy.site.RandomSearch()
	# By default, don't vary the temperature.
	agent = graphity.agent.det.ForwardAgent(ConstBeta(beta), ss)	
	# Use a simulator with metropolis-hastings acceptance to preserve detailed-balance.
	return graphity.task.GlassTask(
		agent=agent, env=graphity.environment.lattice.RejectionSimulator(glass_shape=glass_shape, H=H),
		# The number of toggles is equal to twice the product of the dimension of the glass
		episode_length=functools.reduce(lambda prod,item: prod*item, glass_shape,2),
		name = name,
		number = index,
		sampler = random_sampler,
		trajectories=1)

def create_ground_task(index, beta, glass_shape, name="Ground Search",
	H = graphity.environment.lattice.IsingHamiltonian()):
	"""
	Create a task which will search for minimum energy latticies.
	"""
	random_sampler = graphity.task.RandomGlassSampler(glass_shape)
	ss = graphity.strategy.site.RandomSearch()
	# By default, don't vary the temperature.
	agent = graphity.agent.det.ForwardAgent(ConstBeta(beta), ss)
	# Don't perform conditional acceptance in simulator. Minima may require going through many "bad" states.
	return graphity.task.GlassTask(
		agent=agent, env=graphity.environment.lattice.SpinGlassSimulator(glass_shape=glass_shape, H=H),
		# The number of toggles is equal to twice the product of the dimension of the glass
		episode_length=functools.reduce(lambda prod,item: prod*item, glass_shape,2),
		name = name,
		number = index,
		sampler = random_sampler,
		trajectories=1)
