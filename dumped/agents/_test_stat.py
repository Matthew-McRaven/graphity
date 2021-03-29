import librl.train.train_loop, librl.train.cc.pg

from .. import *
# Test that statistical agents can run without crashing.

def test_forward_agent(GraphSimulator, ForwardAgent, Hypers):
    dist = create_task(GraphSimulator, ForwardAgent, Hypers)
    librl.train.train_loop.cc_episodic_trainer(Hypers, dist, librl.train.cc.policy_gradient_step)
def test_metropolis_agent(GraphSimulator, MetropolisAgent, Hypers):

    dist = create_task(GraphSimulator, MetropolisAgent, Hypers)
    librl.train.train_loop.cc_episodic_trainer(Hypers, dist, librl.train.cc.policy_gradient_step)

def test_forward_agent(GraphSimulator, SimulatedAnnealingAgent, Hypers):
    dist = create_task(GraphSimulator, SimulatedAnnealingAgent, Hypers)
    librl.train.train_loop.cc_episodic_trainer(Hypers, dist, librl.train.cc.policy_gradient_step)
	