from tests.conftest import GraphSimulator
import librl.agent.mdp, librl.agent.pg
import librl.nn.core, librl.nn.actor, librl.nn.critic, librl.nn.pg_loss
import pytest

import graphity
import graphity.agent.core
import graphity.strategy.base, graphity.strategy.grad
@pytest.fixture()
def GradFollower(GraphSimulator):
    return graphity.strategy.grad.TrueGrad(GraphSimulator.H)

@pytest.fixture(params=
    [graphity.strategy.base.random_sampling_strategy,
     graphity.strategy.grad.gd_sampling_strategy,
     graphity.strategy.grad.softmax_sampling_strategy,
     graphity.strategy.grad.beta_sampling_strategy ]
)
def ForwardAgent(GraphSimulator, request):
    SamplingStrategy = request.param
    grad_fn = graphity.strategy.grad.TrueGrad(GraphSimulator.H)
    return graphity.agent.core.ForwardAgent(SamplingStrategy(grad_fn=grad_fn))

@pytest.fixture(params=
    [graphity.strategy.base.random_sampling_strategy,
     graphity.strategy.grad.gd_sampling_strategy,
     graphity.strategy.grad.softmax_sampling_strategy,
     graphity.strategy.grad.beta_sampling_strategy ]
)
def MetropolisAgent(GraphSimulator, request):
    SamplingStrategy = request.param
    grad_fn = graphity.strategy.grad.TrueGrad(GraphSimulator.H)
    return graphity.agent.core.MetropolisAgent(SamplingStrategy(grad_fn=grad_fn))

@pytest.fixture(params=
    [graphity.strategy.base.random_sampling_strategy,
     graphity.strategy.grad.gd_sampling_strategy,
     graphity.strategy.grad.softmax_sampling_strategy,
     graphity.strategy.grad.beta_sampling_strategy ]
)
def SimulatedAnnealingAgent(GraphSimulator, request):
    SamplingStrategy = request.param
    grad_fn = graphity.strategy.grad.TrueGrad(GraphSimulator.H)
    return graphity.agent.core.SimulatedAnnealingAgent(SamplingStrategy(grad_fn=grad_fn), .75, 5, 10)

@pytest.fixture()
def RandomAgent(GraphSimulator):
    return librl.agent.mdp.RandomAgent(GraphSimulator.observation_space, GraphSimulator.action_space)

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