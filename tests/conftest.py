"""
This program demonstrates how to train an agent end-to-end.
It demonstrates all available agents.
It also shows how to create a task distribution to sample from.
"""
import pytest

import graphity.environment.reward, graphity.environment.sim

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
    H = graphity.environment.reward.LogASquaredD(2)
    env = graphity.environment.sim.Simulator(graph_size=request.param, H=H)
    return env