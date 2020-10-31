"""
This file implements policy-gradient based agents.
These agents range from vanila policy gradient (aka REINFORCE)[1] to
proximal policy optimization [2].

For specifications of individual loss functions, see graphity.nn.update_rules

[1] Policy Gradient Methods for Reinforcement Learning with Function Approximation. Sutton et al.
[2] Proximal Policy Optimization Algorithms, Schulman et al.
"""
import torch
import torch.nn as nn
import torch.optim

import graphity.agent
import graphity.nn.update_rules as losses
import graphity.replay


# It caches the last generated policy in self.policy_latest, which can be sampled for additional actions.
@graphity.agent.add_agent_attr(allow_update=True, policy_based=True)
class REINFORCEAgent(nn.Module):
    def __init__(self, hypers, policy_net):
        super(REINFORCEAgent, self).__init__()
        # Cache the last generated policy, so that we can sample for additional actions.
        self.policy_latest = None
        self.policy_net = policy_net
        self.policy_loss = losses.VPG()
        self.policy_optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=hypers['alpha'], weight_decay=hypers['l2'])
        self.hypers = hypers

    def act(self, state):
        return self(state)

    def forward(self, state):
        # Don't return policy information, so as to conform with stochastic agents API.
        actions, logprobs, self.policy_latest = self.policy_net(state)
        return actions, logprobs

    def update(self, state_buffer, action_buffer, reward_buffer, policy_buffer):
        for i in range(1):
            self.policy_optimizer.zero_grad()
            policy_loss = self.policy_loss(state_buffer, action_buffer, reward_buffer, policy_buffer)
            policy_loss.backward()
            self.policy_optimizer.step()

# Implement a common framework for all synchronous actor-critic methods.
# It achieves this versatility by allowing you to specify the policy loss
# function, enabling policy gradient with baseline and PPO to use the same agent.
# You, the user, are responsible for supplying a policy network and value network
# that make sense for the problem.
# It caches the last generated policy in self.policy_latest, which can be sampled for additional actions.
@graphity.agent.add_agent_attr(allow_update=True, policy_based=True)
class ActorCriticAgent(nn.Module):
    def __init__(self, hypers,value_net, policy_net, policy_loss):
        super(ActorCriticAgent, self).__init__()
        # Trust that the caller gave a reasonable value network.
        self.value_net = value_net
        # Goal of our value network (aka the critic) is to make our actual and expected values be equal.
        self.value_loss = torch.nn.MSELoss(reduction="sum")
        # TODO: Optimize with something other than ADAM.
        self.value_optimizer = torch.optim.Adam(self.value_net.parameters(), lr=hypers['alpha'], weight_decay=hypers['l2'])

        self.policy_latest = None
        self.policy_net = policy_net
        self.policy_loss = policy_loss
        # TODO: Optimize with something other than ADAM.
        self.policy_optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=hypers['alpha'], weight_decay=hypers['l2'])
        self.hypers = hypers

    def act(self, state):
        return self(state)

    def value(self, state):
        return self.value_net(state)

    def forward(self, state):
        # Don't return policy information, so as to conform with stochastic agents API.
        actions, logprobs, self.policy_latest = self.policy_net(state)
        return actions, logprobs

    def update(self, state_buffer, action_buffer, reward_buffer, policy_buffer):
        # Train value network.
        # Papers suggest training the critic more often than the actor network.
        for i in range(20):
            self.value_optimizer.zero_grad()
            states = state_buffer.states
            states = states.view(*(states.shape[0:2]),-1)
            value_loss = self.value_loss(self.value_net(states), reward_buffer.rewards)
            #print(f"Value loss: {value_loss.item()}")
            value_loss.backward()
            self.value_optimizer.step()

        # Train policy network.
        for i in range(1):
            self.policy_optimizer.zero_grad()
            policy_loss = self.policy_loss(state_buffer, action_buffer, reward_buffer, policy_buffer)
            policy_loss.backward()
            self.policy_optimizer.step()