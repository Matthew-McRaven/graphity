from tests.conftest import GraphSimulator
import librl.agent.mdp, librl.agent.pg
import librl.nn.core, librl.nn.actor, librl.nn.critic, librl.nn.pg_loss
import pytest

import graphity
import graphity.agent.core
import graphity.strategy.base, graphity.strategy.grad
import graphity.environment.reward

from .. import *

@pytest.fixture(params=
    [
        graphity.environment.reward.IsingHamiltonian(),
        graphity.environment.reward.SpinGlassHamiltonian(categorical=False),
        graphity.environment.reward.SpinGlassHamiltonian(categorical=True)
    ]
)
def SpinHamiltonianSimulator(request):
    H = request.param
    env = graphity.environment.sim.Simulator(graph_size=5, H=H)
    return env

@pytest.fixture()
def ForwardAgent(SpinHamiltonianSimulator):
    sampling_strategy = graphity.strategy.base.random_sampling_strategy
    return graphity.agent.core.ForwardAgent(sampling_strategy())