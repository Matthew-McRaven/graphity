"""
This program demonstrates how to train an agent end-to-end.
It demonstrates all available agents.
It also shows how to create a task distribution to sample from.
"""
import torch
import librl.nn.core, librl.nn.actor, librl.nn.critic, librl.nn.pg_loss
import librl.agent.pg, librl.agent.mdp
import librl.task.distribution, librl.task
import librl.train.train_loop, librl.train.cc.pg

import graphity.agent.markov, graphity.agent.grad
import graphity.grad
import graphity.environment.reward, graphity.environment.sim
import graphity.hypers, graphity.train
import graphity.task

# Sample program that demonstrates how to create an agent & environment.
# Then. train this agent for some number of epochs, determined by our hypers.
def main():
    hypers = graphity.hypers.get_default_hyperparams()

    hypers['epochs'] = 4
    hypers['episode_count'] = 4
    hypers['task_count'] = 1
    hypers['episode_length'] = 100
    hypers['graph_size'] = 6
    hypers['toggles_per_step'] = 2

    # Environment definition
    H = graphity.environment.reward.LogASquaredD(2)
    env = graphity.environment.sim.Simulator(graph_size=hypers['graph_size'], H=H)

    # Stochastic agents
    #agent = librl.agent.mdp.RandomAgent(env.observation_space, env.action_space)
    #agent = graphity.agent.markov.MDPAgent()

    # Gradient descent agents
    agent = graphity.agent.grad.GradientFollowingAgent(H)
    
    # Neural-network based agents
    value_kernel = librl.nn.core.MLPKernel(hypers['graph_size']**2, [211])
    value_net = librl.nn.critic.ValueCritic(value_kernel)
    policy_kernel = librl.nn.core.MLPKernel(hypers['graph_size']**2, [117])
    policy_net = librl.nn.actor.BiCategoricalActor(policy_kernel, env.action_space, env.observation_space)
    # Vanilla policy gradient
    #agent = librl.agent.pg.REINFORCEAgent(policy_net)
    # Actor-critic policy gradient methods.
    # Change policy loss fn to change behavior of agent.
    #policy_loss = librl.nn.pg_loss.PGB(value_net)
    #policy_loss = librl.nn.pg_loss.PPO(value_net)
    #agent = librl.agent.pg.ActorCriticAgent(value_net, policy_net, policy_loss)

    # Show the NN configuration on the console.
    print(agent)

    # Define different sampling methods for points
    random_sampler = graphity.task.RandomSampler(hypers['graph_size'])
    #checkpoint_sampler = graphity.task.CheckpointSampler(random_sampler) # Suspiciously wrong.
    dist = librl.task.TaskDistribution()
    # Create a single task definition from which we can sample.
    dist.add_task(librl.task.Task.Definition(graphity.task.GraphTask, sampler=random_sampler, agent=agent, env=env))

    librl.train.train_loop.cc_episodic_trainer(hypers, dist, librl.train.cc.policy_gradient_step)


if __name__ == "__main__":
    main()