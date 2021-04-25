import functools

import graphity.agent.det
import graphity.environment.lattice
import graphity.task
import graphity.strategy.site
class const_temp:
	def __init__(self, beta):
		self.beta = beta
	def __call__(self, beta, delta_e):
		return self.beta, 0
def create_eq_task(index, beta, glass_shape, name="Lingus"):
	H = graphity.environment.lattice.IsingHamiltonian()
	random_sampler = graphity.task.RandomGlassSampler(glass_shape)
	ss = graphity.strategy.site.RandomSearch()
	agent = graphity.agent.det.ForwardAgent(const_temp(beta), ss)	
	return graphity.task.GlassTask(
		agent=agent, env=graphity.environment.lattice.RejectionSimulator(glass_shape=glass_shape, H=H), 
		episode_length=functools.reduce(lambda prod,item: prod*item, glass_shape,2),
		name = name,
		number = index,
		sampler = random_sampler,
		trajectories=1)

def create_ground_task(index, beta, glass_shape, name="Lingus"):
	H = graphity.environment.lattice.IsingHamiltonian()
	random_sampler = graphity.task.RandomGlassSampler(glass_shape)
	ss = graphity.strategy.site.RandomSearch()
	agent = graphity.agent.det.ForwardAgent(const_temp(beta), ss)	
	return graphity.task.GlassTask(
		agent=agent, env=graphity.environment.lattice.SpinGlassSimulator(glass_shape=glass_shape, H=H), 
		episode_length=functools.reduce(lambda prod,item: prod*item, glass_shape,2),
		name = name,
		number = index,
		sampler = random_sampler,
		trajectories=1)
