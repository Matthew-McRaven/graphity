import librl.train.train_loop, librl.train.cc.pg

from .. import *

def test_all_models(SpinHamiltonianSimulator, ForwardAgent, Hypers):
    dist = create_task(SpinHamiltonianSimulator, ForwardAgent, Hypers)
    librl.train.train_loop.cc_episodic_trainer(Hypers, dist, librl.train.cc.policy_gradient_step)
