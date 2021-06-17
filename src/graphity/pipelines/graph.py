import functools

import graphity.agent.det
import graphity.environment.lattice
from graphity.strategy.anneal import ConstBeta
import graphity.strategy.site
import graphity.task


##################################
# Helpers to create graph tasks. #
##################################

def create_eq_task(index, beta, graph_shape, H, name="EQ",):
	"""
	Create a task which will evolve random lattices via a random walk and metropolis-hastings acceptance.

	:param index: A unique numeric identifier for this task. Only needs to be unique between tasks being run in the same pipeline, not globally.
	:param beta: Inverse temperature of the system being simulated.
	:param graph_shape: Dimensions of the spin glass being simulated. Works for up to 3 dimensions, with numbers of nodes ~10k.
	:param name: A string identifier describing the type of task (e.g., equilibriation vs ground state search.). Defaults to "EQ".
	:param H: The Hamiltonian under which the system will be simulated., defaults to the Ising model Hamiltonian.
	"""
	random_sampler = graphity.task.RandomGlassSampler(graph_shape)
	# Equilibritation must preserve detailed-balance, which means we need to do a random walk.
	ss = graphity.strategy.site.RandomSearch()
	# By default, don't vary the temperature.
	agent = graphity.agent.det.ForwardAgent(ConstBeta(beta), ss)	
	# Use a simulator with metropolis-hastings acceptance to preserve detailed-balance.
	return graphity.task.GlassTask(
		agent=agent, env=graphity.environment.graph.RejectionSimulator(graph_shape=graph_shape, H=H),
		# The number of toggles is equal to twice the product of the dimension of the glass
		episode_length=functools.reduce(lambda prod,item: prod*item, graph_shape,2),
		name = name,
		number = index,
		sampler = random_sampler,
		trajectories=1)

def to_spin(state):
	state[state==0] = -1
	return state