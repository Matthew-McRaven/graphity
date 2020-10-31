"""
This program demonstrates how to train an agent end-to-end.
It can be used to track the progress of multiple agents on the same task.
"""
import torch

import graphity.agent.markov, graphity.agent.grad, graphity.agent.pg
import graphity.nn.actor, graphity.nn.critic, graphity.nn.update_rules
import graphity.grad
import graphity.replay
import graphity.environment.reward, graphity.environment.sim
import graphity.hypers, graphity.train

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

    graphity.train.simulate_epoch(hypers, agent, env)


if __name__ == "__main__":
    main()