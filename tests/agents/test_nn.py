import librl.train.train_loop, librl.train.cc.pg

from .. import *
# Test that REINFORCE, AC agents can run on graph without crashing.

def test_reinforce_agent(GraphSimulator, REINFORCEAgent, Hypers):
    dist = create_task(GraphSimulator, REINFORCEAgent, Hypers)
    librl.train.train_loop.cc_episodic_trainer(Hypers, dist, librl.train.cc.policy_gradient_step)

def test_ac_agent(GraphSimulator, ActorCriticAgent, Hypers):
	dist = create_task(GraphSimulator, ActorCriticAgent, Hypers)
	librl.train.train_loop.cc_episodic_trainer(Hypers, dist, librl.train.cc.policy_gradient_step)