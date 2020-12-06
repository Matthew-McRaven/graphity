import librl.train.train_loop, librl.train.cc.pg

from . import *
# Test that statistical agents can run without crashing.

def test_random_agent(GraphSimulator, RandomAgent, Hypers):
    dist = create_task(GraphSimulator, RandomAgent, Hypers)
    librl.train.train_loop.cc_episodic_trainer(Hypers, dist, librl.train.cc.policy_gradient_step)

def test_grad_agent(GraphSimulator, GradFollowingAgent, Hypers):
	dist = create_task(GraphSimulator, GradFollowingAgent, Hypers)
	librl.train.train_loop.cc_episodic_trainer(Hypers, dist, librl.train.cc.policy_gradient_step)

def test_lookback_agent(GraphSimulator, GradFollowingAgent, Hypers):
	dist = create_task(GraphSimulator, GradFollowingAgent, Hypers)
	librl.train.train_loop.cc_episodic_trainer(Hypers, dist, librl.train.cc.policy_gradient_step)