"""
This program demonstrates how to train an agent end-to-end.
It demonstrates all available agents.
It also shows how to create a task distribution to sample from.
"""
import torch

import graphity.agent.mdp, graphity.agent.pg, graphity.agent.det
import graphity.grad
import graphity.environment.lattice
import graphity.environment.biqg
import graphity.train
import graphity.task
import graphity.train.ground
import graphity.strategy
import graphity.train
# Sample program that demonstrates how to create an agent & environment.
# Then. train this agent for some number of epochs, determined by our hypers.
def main():
    hypers = {} 

    hypers['epochs'] = 4
    hypers['episode_count'] = 1
    hypers['task_count'] = 1
    hypers['episode_length'] = 10000
    hypers['graph_size'] = 20
    hypers['toggles_per_step'] = 1

    # Environment definition
    #H = graphity.environment.biqg.ASquaredD(2)
    #env = graphity.environment.biqg.Simulator(H, graph_size=hypers['graph_size'])
    H = graphity.environment.lattice.IsingHamiltonian()
    glass_shape = (hypers['graph_size'],hypers['graph_size'])
    env = graphity.environment.lattice.SpinGlassSimulator(glass_shape=glass_shape, H=H)
    hypers['env'] = env
    # Stochastic agents
    # Gradient descent agents
    ss = graphity.strategy.RandomSearch()

    agents = []
    agents.append(("fa", graphity.agent.det.ForwardAgent(ss)))
    agents.append(("ma", graphity.agent.mdp.MetropolisAgent(ss)))
    agents.append(("sa", graphity.agent.mdp.SimulatedAnnealingAgent(ss, 2, 4, 1)))

    # Show the NN configuration on the console.
    print(agents)

    random_sampler = graphity.task.RandomSampler(hypers['graph_size'])
    dist = graphity.task.TaskDistribution()
    # Create a single task definition from which we can sample.
    for (idx, (name, agent)) in enumerate(agents):
        dist.add_task(graphity.task.Definition(graphity.task.GraphTask, 
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