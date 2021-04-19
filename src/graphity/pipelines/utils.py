import functools

import graphity.agent.det
import graphity.environment.lattice
import graphity.task
import graphity.strategy.site
def const_temp(beta, delta_e):
	return beta, 0
def create_task(index, beta, glass_shape, name="Lingus"):
	H = graphity.environment.lattice.IsingHamiltonian()
	random_sampler = graphity.task.RandomGlassSampler(glass_shape)
	ss = graphity.strategy.site.RandomSearch()
	agent = graphity.agent.det.ForwardAgent(const_temp, ss)	
	return graphity.task.GlassTask(
		agent=agent, env=graphity.environment.lattice.RejectionSimulator(glass_shape=glass_shape, H=H), 
		episode_length=functools.reduce(lambda prod,item: prod*item, glass_shape,2),
		name = name,
		number = index,
		sampler = random_sampler,
		trajectories=1)
