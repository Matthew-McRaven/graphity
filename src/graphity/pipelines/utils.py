import functools

import graphity.agent.det
import graphity.environment.lattice
import graphity.task
import graphity.strategy.site

def create_task(index, beta, glass_shape, name="Lingus"):
	H = graphity.environment.lattice.IsingHamiltonian()
	random_sampler = graphity.task.RandomGlassSampler(glass_shape)
	ss = graphity.strategy.site.RandomSearch()
	agent = graphity.agent.det.ForwardAgent(lambda x,y:(beta,0), ss)	
	return graphity.task.GlassTask(
		agent=agent, env=graphity.environment.lattice.RejectionSimulator(glass_shape=glass_shape, H=H), 
		episode_length=functools.reduce(lambda prod,item: prod*item, glass_shape,2),
		name = name,
		number = index,
		sampler = random_sampler,
		trajectories=1)

def var(batch):
	summed = functools.reduce(lambda sum, item: sum + item, batch, 0)
	squared_sums = functools.reduce(lambda sum, item: sum + item**2, batch,0)
	return squared_sums/len(batch) - (summed/len(batch))**2