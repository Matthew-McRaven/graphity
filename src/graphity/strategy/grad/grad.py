import torch
import torch.nn as nn
import numpy as np

# Needed to compute grads
import graphity.grad

from .utils import *

class gd_sampling_strategy:
    def __init__(self, grad_fn=None, mask_triu=True):
        assert grad_fn
        self.grad_fn = grad_fn
        self.mask_triu = mask_triu

    def __call__(self, adj):
        size = adj.shape[-1]
        grad = self.grad_fn(adj)
        
        masked_grad = mask_grads(grad) if self.mask_triu else grad
        
        # Pick top transition than most minimizes energy.
        _, index = torch.topk(masked_grad.view(-1), 1, largest=False)
        picked = np.unravel_index(index.item(), masked_grad.shape)
        action = torch.tensor(picked[0]).view(-1)
        site = torch.tensor(picked[1:]).view(-1)

        # Must stack along dim=-1 in order to properly join pairs.
        actions = torch.cat((site, action+2), dim=-1).to(adj.device)
        return actions, torch.zeros((1,), device=adj.device)

class softmax_sampling_strategy:
    def __init__(self, grad_fn=None, mask_triu=True): 
        assert grad_fn
        self.grad_fn = grad_fn
        self.mask_triu = mask_triu

    def __call__(self, adj):
        size = adj.shape[-1]
        grad = self.grad_fn(adj)
        masked_grad = mask_grads(grad) if self.mask_triu else grad

        probs = torch.softmax(masked_grad.view(-1), 0).view(-1)

        dist = torch.distributions.categorical.Categorical(probs)
        index = dist.sample((1,))
        picked = np.unravel_index(index.item(), masked_grad.shape)
        action = torch.tensor(picked[0]).view(-1)
        site = torch.tensor(picked[1:]).view(-1)

        # Must stack along dim=-1 in order to properly join pairs.
        actions = torch.cat((site,action+2), dim=-1).to(adj.device)
        """print("SMGD")
        print(adj)
        print(grad)
        print(index)
        print(actions)
        print()"""
        return actions, dist.log_prob(index)


class beta_sampling_strategy:
    def __init__(self, grad_fn=None, alpha=.5, beta=2, mask_triu=True):
        assert grad_fn
        self.grad_fn = grad_fn
        # Grads now in [0,1], so use beta dist to biasedly pick "best" transition.
        self.dist = torch.distributions.beta.Beta(alpha, beta)
        self.mask_triu = mask_triu

    def __call__(self, adj):
        size = adj.shape[-1]
        grad = self.grad_fn(adj)
        min, max = torch.min(grad.view(-1)), torch.max(grad.view(-1))
        grad = (grad - min)/ (max - min)
        #TODO: Range compression.
        masked_grad = mask_grads(grad) if self.mask_triu else grad

        value = self.dist.sample((1,))
        index = torch.argmin((masked_grad-value).abs())
        picked = np.unravel_index(index.item(), masked_grad.shape)
        action = torch.tensor(picked[0]).view(-1)
        site = torch.tensor(picked[1:]).view(-1)

        # Must stack along dim=-1 in order to properly join pairs.
        actions = torch.cat((site,action+2), dim=-1).to(adj.device)
        """print("BGD")
        print(adj)
        print(grad)
        print(index, value)
        print(actions)
        print()"""
        return actions, self.dist.log_prob(value)
