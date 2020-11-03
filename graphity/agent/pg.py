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
    def __init__(self, hypers, actor_net):
        super(REINFORCEAgent, self).__init__()
        # Cache the last generated policy, so that we can sample for additional actions.
        self.policy_latest = None
        self.actor_net = actor_net
        self._actor_loss = losses.VPG()
        self.actor_optimizer = torch.optim.Adam(self.actor_net.parameters(), lr=hypers['alpha'], weight_decay=hypers['l2'])
        self.hypers = hypers

    def act(self, state, toggles=1):
        return self(state, toggles)

    def forward(self, state, toggles=1):
        # Don't return policy information, so as to conform with stochastic agents API.
        actions, logprobs, self.policy_latest = self.actor_net(state, toggles)
        return actions, logprobs

    def actor_loss(self, task):
        return self._actor_loss(task.state_buffer, task.action_buffer, task.reward_buffer, task.policy_buffer)

# Implement a common framework for all synchronous actor-critic methods.
# It achieves this versatility by allowing you to specify the policy loss
# function, enabling policy gradient with baseline and PPO to use the same agent.
# You, the user, are responsible for supplying a policy network and value network
# that make sense for the problem.
# It caches the last generated policy in self.policy_latest, which can be sampled for additional actions.
@graphity.agent.add_agent_attr(allow_update=True, policy_based=True)
class ActorCriticAgent(nn.Module):
    def __init__(self, hypers, critic_net, actor_net, actor_loss):
        super(ActorCriticAgent, self).__init__()
        self.hypers = hypers

        # Trust that the caller gave a reasonable value network.
        self.critic_net = critic_net
        # Goal of our value network (aka the critic) is to make our actual and expected values be equal.
        self._critic_loss = torch.nn.MSELoss(reduction="sum")
        # TODO: Optimize with something other than ADAM.
        self.critic_optimizer = torch.optim.Adam(self.critic_net.parameters(), lr=hypers['alpha'], weight_decay=hypers['l2'])

        self.policy_latest = None
        self.actor_net = actor_net
        self._actor_loss = actor_loss
        # TODO: Optimize with something other than ADAM.
        self.actor_optimizer = torch.optim.Adam(self.actor_net.parameters(), lr=hypers['alpha'], weight_decay=hypers['l2'])

    def act(self, state, toggles=1):
        return self(state, toggles)

    def value(self, state):
        return self.critic_net(state)

    def forward(self, state, toggles=1):
        # Don't return policy information, so as to conform with stochastic agents API.
        actions, logprobs, self.policy_latest = self.actor_net(state, toggles)
        return actions, logprobs
        
    def actor_loss(self, task):
        return self._actor_loss(task.state_buffer, task.action_buffer, task.reward_buffer, task.policy_buffer)

    def critic_loss(self, task):
        # Must reshape states to be a batched 1d array.
        # TODO: Will need different reshaping for CNN's.
        states = task.state_buffer.states
        states = states.view(*(states.shape[0:2]),-1)
        return self._critic_loss(self.critic_net(states), task.reward_buffer.rewards)
