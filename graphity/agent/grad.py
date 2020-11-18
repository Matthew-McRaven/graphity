import torch
import torch.nn as nn

# Needed for add_agent_attr() decorator
import librl.agent
# Needed to compute grads
import graphity.grad

# The random agent random selects one edge pair to toggle per timestep.
@librl.agent.add_agent_attr()
class GradientFollowingAgent(nn.Module):
    def __init__(self, H, hypers):
        # Must initialize torch.nn.Module
        super(GradientFollowingAgent, self).__init__()
        self.H = H

    def act(self, adj, toggles=1):
        return self.forward(adj, toggles)

    # Implement required pytorch interface
    def forward(self, adj, toggles=1):
        size = adj.shape[-1]
        grad = graphity.grad.graph_gradient(adj, self.H).tril()

        # Prevent NN from toggling diagonal and duplicate edges in upper tril.
        upper_inf = torch.full((size,size), float('inf')).triu(0)
        grad = (grad + upper_inf).view(-1)

        # Pick k transitions than minimize energy the most.
        _, indicies = torch.topk(grad, toggles, largest=False)

        # Contains all of the columns
        first = indicies // (size)
        # Contains all of the rows
        second = indicies % (size)
        
        # Must stack along dim=-1 in order to properly join pairs.
        actions = torch.stack([first,second], dim=-1)

        return actions, torch.zeros((1,))