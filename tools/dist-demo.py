"""
This program demonstrates how to train an agent end-to-end.
It demonstrates all available agents.
It also shows how to create a task distribution to sample from.
"""
import torch

import graphity.agent.det
import graphity.environment.lattice
import graphity.train
import graphity.task
import graphity.train.ground
import graphity.site_strategy
import graphity.train
# Sample program that demonstrates how to create an agent & environment.
# Then. train this agent for some number of epochs, determined by our hypers.
def main():
    hypers = {} 

    hypers['epochs'] = 4
    hypers['episode_count'] = 1
    hypers['task_count'] = 1
    hypers['episode_length'] = 1000
    hypers['graph_size'] = 4
    hypers['toggles_per_step'] = 1

    # Environment definition
    #H = graphity.environment.biqg.ASquaredD(2)
    #env = graphity.environment.biqg.Simulator(H, graph_size=hypers['graph_size'])
    H = graphity.environment.lattice.IsingHamiltonian()
    #H = graphity.environment.lattice.SpinGlassHamiltonian(categorical=True)
    glass_shape = (hypers['graph_size'], hypers['graph_size'])
    env = graphity.environment.lattice.SpinGlassSimulator(glass_shape=glass_shape, H=H)
    hypers['env'] = env
    # Stochastic agents
    # Gradient descent agents
    ss = graphity.site_strategy.RandomSearch()
    nb = graphity.site_strategy.Score1Neighbors(H)
    gd = graphity.site_strategy.VanillaILS(nb)
    smgd = graphity.site_strategy.SoftmaxILS(nb)
    bgd = graphity.site_strategy.BetaILS(nb)
    agents = []
    agents.append(("gd", graphity.agent.det.ForwardAgent(lambda x,y:(1,0), gd)))
    agents.append(("sm", graphity.agent.det.ForwardAgent(lambda x,y:(1,0), smgd)))
    agents.append(("bgd", graphity.agent.det.ForwardAgent(lambda x,y:(1,0), bgd)))
    agents.append(("fa", graphity.agent.det.ForwardAgent(lambda x,y:(1,0), ss)))
    #agents.append(("ma", graphity.agent.mdp.MetropolisAgent(ss)))
    #agents.append(("sa", graphity.agent.mdp.SimulatedAnnealingAgent(ss, 2, 4, 1)))

    # Show the NN configuration on the console.
    print(agents)

    random_sampler = graphity.task.RandomGlassSampler(glass_shape)
    graph = graphity.environment.lattice.random_glass(glass_shape)
    dist = graphity.task.TaskDistribution()
    # Create a single task definition from which we can sample.
    for (idx, (name, agent)) in enumerate(agents):
        dist.add_task(graphity.task.Definition(graphity.task.GlassTask, 
            agent=agent, env=hypers['env'], 
            episode_length=hypers['episode_length'],
            name = name,
            number = idx,
            sampler = random_sampler,
            trajectories=1)
        )
    logger = graphity.train.DirLogger(H, "./logs.zip")
    graphity.train.ground.train_ground_search(hypers, dist, lambda x:x, logger)


if __name__ == "__main__":
   main()