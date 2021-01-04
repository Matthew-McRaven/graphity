import librl.agent.mdp, librl.agent.pg
import librl.nn.core, librl.nn.actor, librl.nn.critic, librl.nn.pg_loss
import pytest

import graphity.agent.mdp, graphity.agent.grad

@pytest.fixture()
def RandomAgent(GraphSimulator):
    return librl.agent.mdp.RandomAgent(GraphSimulator.observation_space, GraphSimulator.action_space)

@pytest.fixture()
def GradFollowingAgent(GraphSimulator):
    return graphity.agent.grad.GradientFollowingAgent(GraphSimulator.H)

@pytest.fixture()
def GradFollowingAgent(GraphSimulator):
    return graphity.agent.mdp.MetropolisMarkovAgent()

@pytest.fixture()
def REINFORCEAgent(GraphSimulator):
    policy_kernel = librl.nn.core.MLPKernel(GraphSimulator.graph_size**2, [117])
    policy_net = librl.nn.actor.BiCategoricalActor(policy_kernel, GraphSimulator.action_space, GraphSimulator.observation_space)
    return librl.agent.pg.REINFORCEAgent(policy_net)

@pytest.fixture(params=[librl.nn.pg_loss.PGB, librl.nn.pg_loss.PPO])
def ActorCriticAgent(GraphSimulator, request):
	# Neural-network based agents
    value_kernel = librl.nn.core.MLPKernel(GraphSimulator.graph_size**2, [211])
    value_net = librl.nn.critic.ValueCritic(value_kernel)
    policy_kernel = librl.nn.core.MLPKernel(GraphSimulator.graph_size**2, [117])
    policy_net = librl.nn.actor.BiCategoricalActor(policy_kernel, GraphSimulator.action_space, GraphSimulator.observation_space)
    policy_loss = request.param(value_net)
    return librl.agent.pg.ActorCriticAgent(value_net, policy_net, policy_loss)