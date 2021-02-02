from .grad import *
import graphity.grad
# Compute the true gradient, rather than using an approximation.
class TrueGraphGrad:
    def __init__(self, H):
        assert H
        self.H = H
    def __call__(self, adj):
        return graph_gradient(adj, self.H)

class TrueSpinGrad:
    def __init__(self, H, action_count):
        assert H
        self.H = H
        self.action_count = action_count
    def __call__(self, adj):
        return graphity.grad.spin_gradient(adj, self.H, self.action_count)

# Approximate the gradient using a neural network.
class NeuralGrad:
    def __init__(self, model):
        self.model = model
    def __call__(self, adj):
        return self.model(adj)

