"""
This program demonstrates how to train an agent end-to-end.
It demonstrates all available agents.
It also shows how to create a task distribution to sample from.
"""
import torch

import graphity.agent.markov, graphity.agent.grad, graphity.agent.pg
import graphity.nn.actor, graphity.nn.critic, graphity.nn.update_rules
import graphity.grad
import graphity.replay
import graphity.environment.reward, graphity.environment.sim
import graphity.hypers, graphity.train
import graphity.task

# Sample program that demonstrates how to create an agent & environment.
# Then. train this agent for some number of epochs, determined by our hypers.
def main():
    hypers = graphity.hypers.get_default_hyperparams()

    hypers['epochs'] = 4
    hypers['episode_count'] = 4
    hypers['episode_length'] = 100
    hypers['graph_size'] = 6
    hypers['toggles_per_step'] = 2

    # Environment definition
    H = graphity.environment.reward.LogASquaredD(2)
    env = graphity.environment.sim.Simulator(graph_size=hypers['graph_size'], H=H)

    # Stochastic agents
    #agent = graphity.agent.markov.RandomAgent(hypers)
    #agent = graphity.agent.markov.MDPAgent(hypers)

    # Gradient descent agents
    agent = graphity.agent.grad.GradientFollowingAgent(H, hypers)
    
    # Neural-network based agents
    value_net = graphity.nn.critic.MLPCritic(hypers['graph_size']**2, hypers)
    policy_net = graphity.nn.actor.MLPActor(hypers['graph_size']**2, hypers)
    # Vanilla policy gradient
    #agent = graphity.agent.pg.REINFORCEAgent(hypers, policy_net)
    # Actor-critic policy gradient methods.
    # Change policy loss fn to change behavior of agent.
    #policy_loss = graphity.nn.update_rules.PGB(value_net, hypers)
    policy_loss = graphity.nn.update_rules.PPO(value_net, hypers)
    #agent = graphity.agent.pg.ActorCriticAgent(hypers, value_net, policy_net, policy_loss)

    # Show the NN configuration on the console.
    print(agent)

    # Define different sampling methods for points
    random_sampler = graphity.task.RandomSampler(hypers['graph_size']**2)
    checkpoint_sampler = graphity.task.CheckpointSampler(random_sampler) # Suspiciously wrong.
    dist = graphity.task.TaskDistribution()
    # Create a single task definition from which we can sample.
    dist.add_task(graphity.task.TaskDefinition(checkpoint_sampler, policy_loss, hypers))

    graphity.train.basic_task_loop(hypers, env, agent, dist)


if __name__ == "__main__":
    main()