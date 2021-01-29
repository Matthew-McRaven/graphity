# Compute the true gradient, rather than using an approximation.
class TrueGrad:
    def __init__(self, H):
        assert H
        self.H = H
    def __call__(self, adj):
        return graphity.grad.graph_gradient(adj, self.H)

# Approximate the gradient using a neural network.
class NeuralGrad:
    def __init__(self, model):
        self.model = model
    def __call__(self, adj):
        return self.model(adj)

