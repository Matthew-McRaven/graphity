import torch
import torch.nn as nn

# Needed for add_agent_attr() decorator
import librl.agent
import librl.train.cc.pg
# Needed to compute grads
import graphity.grad

# Compute the true gradient, rather than using an approximation.
class TrueGrad:
    def __init__(self, H):
        self.H = H
    def __call__(self, adj):
        return graphity.grad.graph_gradient(adj, self.H).tril()

# Approximate the gradient using a neural network.
class NeuralGrad:
    def __init__(self, model):
        self.model = model
    def __call__(self, adj):
        return self.model(adj)

def mask_grads(grad):
    size = grad.shape[-1]
    # Prevent NN from toggling diagonal and duplicate edges in upper tril.
    upper_mask = torch.full((size,size), float('inf')).triu(0)
    return -(grad + upper_mask)

class gd_sampling_strategy:
    def __init__(self, grad_fn):
        self.grad_fn = grad_fn

    def __call__(self, adj):
        size = adj.shape[-1]
        grad = self.grad_fn(adj)
        masked_grad = mask_grads(grad)

        # Pick top transition than most minimizes energy.
        _, index = torch.topk(masked_grad.view(-1), 1, largest=True)

        # Convert 1d index to 2d index (i.e., column & row)
        col, row = index // (size),  index % size

        # Must stack along dim=-1 in order to properly join pairs.
        actions = torch.stack([col,row], dim=-1).to(adj.device)
        return actions, torch.tensor(0., device=adj.device)

class softmax_sampling_strategy:
    def __init__(self, grad_fn): 
        self.grad_fn = grad_fn

    def __call__(self, adj):
        size = adj.shape[-1]
        grad = self.grad_fn(adj)
        masked_grad = mask_grads(grad)
        probs = torch.softmax(masked_grad.view(-1), 0).view(-1)
        index, col, row = None,None, None
        dist = torch.distributions.categorical.Categorical(probs)
        while col == row:
            index = dist.sample((1,))
            col = index // (size)
            row = index % size
        
        # Must stack along dim=-1 in order to properly join pairs.
        actions = torch.stack([col,row], dim=-1).to(adj.device)
        return actions, dist.log_prob(index)


class beta_sampling_strategy:
    def __init__(self, grad_fn, alpha=.5, beta=2):
        self.grad_fn = grad_fn
        # Grads now in [0,1], so use beta dist to biasedly pick "best" transition.
        self.dist = torch.distributions.beta.Beta(alpha, beta)

    def __call__(self, adj):
        size = adj.shape[-1]
        grad = self.grad_fn(adj)
        min, max = torch.min(grad.view(-1)), torch.max(grad.view(-1))
        grad = (grad - min)*(-1)/ (max - min)
        #TODO: Range compression.
        masked_grad = mask_grads(grad)
        value = self.dist.sample((1,))
        index = torch.argmin((masked_grad-value).abs())
        # Convert 1d index to 2d index (i.e., column & row)
        col, row = index // (size),  index % size
        # Must stack along dim=-1 in order to properly join pairs.
        actions = torch.stack([col,row], dim=-1).to(adj.device)
        return actions, self.dist.log_prob(value)