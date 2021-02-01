"""
This program demonstrates how to train an agent end-to-end.
It demonstrates all available agents.
It also shows how to create a task distribution to sample from.
"""
import pytest

import graphity.agent.mdp, graphity.agent.pg, graphity.agent.det
import graphity.grad
import graphity.environment.lattice
import graphity.environment.biqg
import graphity.train
import graphity.task
import graphity.train.ground
import graphity.strategy
import graphity.train

@pytest.fixture()
def Hypers():
    hypers = {}
    hypers['epochs'] = 4
    hypers['episode_count'] = 4
    hypers['task_count'] = 1
    hypers['episode_length'] = 100
    return hypers

@pytest.fixture(params=[6, 8])
def GraphSimulator(request):
    H = graphity.environment.biqg.LogASquaredD(2)
    env = graphity.environment.biqg.GraphSimulator(graph_size=request.param, H=H)
    return env

@pytest.fixture(params=[6, 8])
def IsingGlassSimulator(request):
    H = graphity.environment.lattice.IsingHamiltonian() 
    n = request.param
    env = graphity.environment.lattice.SpinGlassSimulator(glass_shape=(n,n), H=H)
    return env