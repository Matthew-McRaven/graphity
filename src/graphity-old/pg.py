"""
This file implements policy-gradient based agents.
These agents range from vanila policy gradient (aka REINFORCE)[1] to
proximal policy optimization [2].

For specifications of individual loss functions, see graphity.nn.update_rules

[1] Policy Gradient Methods for Reinforcement Learning with Function Approximation. Sutton et al.
[2] Proximal Policy Optimization Algorithms, Schulman et al.
"""
import copy
import types
import functools

import torch
import torch.nn as nn
import torch.optim

# Needed for add_agent_attr() decorator
from . import add_agent_attr
# It caches the last generated policy in self.policy_latest, which can be sampled for additional actions.
@add_agent_attr(allow_update=True, policy_based=True)
class REINFORCEAgent(nn.Module):
    def __init__(self, actor_net, gamma=None, explore_bonus_fn=lambda _:0, alpha=None, l2=None, actor_loss_mult=None):
        super(REINFORCEAgent, self).__init__()
        self.alpha = alpha if alpha else self.get_default_hyperparameters().alpha
        self.l2 = l2 if l2 else self.get_default_hyperparameters().l2
        self.actor_loss_mult = actor_loss_mult if actor_loss_mult else self.get_default_hyperparameters().actor_loss_mult
        
        # Cache the last generated policy, so that we can sample for additional actions.
        self.policy_latest = None
        self.actor_net = actor_net
        # Don't overwrite defaults of VPG if we are going to be passing in a None
        self._actor_loss = librl.nn.pg_loss.VPG(**{k:v for (k,v) in {'gamma':gamma, 'explore_bonus_fn':explore_bonus_fn}.items() if v is not None})
        self.actor_optimizer = torch.optim.RMSprop(self.actor_net.parameters(), lr=self.alpha, weight_decay=self.l2)

    @staticmethod
    def get_default_hyperparameters():
        ret = {}
        ret['alpha'] = .0001
        ret['l2'] = 0.2
        ret['actor_loss_mult'] = -1
        return types.SimpleNamespace(**ret)

    def recurrent(self):
        return self.actor_net.recurrent()

    def save_hidden(self):
        assert self.actor_net.recurrent()
        return {id(self.actor_net):self.actor_net.save_hidden()}

    def restore_hidden(self, state_dict=None):
        assert self.actor_net.recurrent()
        id_actor = id(self.actor_net)
        if state_dict == None: self.actor_net.restore_hidden()
        elif id(id_actor) in state_dict: self.actor_net.restore_hidden(state_dict[id_actor])
        else: assert 0 and "Missing keys!"

    def act(self, state):
        return self(state)

    def forward(self, state):
        # Don't return policy information, so as to conform with stochastic agents API.
        actions, logprobs, self.policy_latest = self.actor_net(state)
        return actions, logprobs

    def actor_loss(self, task):
        return self._actor_loss(task)

    def steal(self):
        return copy.deepcopy(self.state_dict())

    def stuff(self, params):
        self.load_state_dict(params, strict=True)

# Implement a common framework for all synchronous actor-critic methods.
# It achieves this versatility by allowing you to specify the policy loss
# function, enabling policy gradient with baseline and PPO to use the same agent.
# You, the user, are responsible for supplying a policy network and value network
# that make sense for the problem.
# It caches the last generated policy in self.policy_latest, which can be sampled for additional actions.
@add_agent_attr(allow_update=True, policy_based=True)
class ActorCriticAgent(nn.Module):
    def __init__(self, critic_net, actor_net, actor_loss, actor_alpha=None, actor_l2=None,
                critic_alpha=None, critic_l2=None, critic_steps=None, actor_loss_mult=None):
        super(ActorCriticAgent, self).__init__()
        self.actor_loss_mult = actor_loss_mult if actor_loss_mult else self.get_default_hyperparameters().actor_loss_mult
        self.actor_alpha = actor_alpha if actor_alpha else self.get_default_hyperparameters().actor_alpha
        self.actor_l2 = actor_l2 if actor_l2 else self.get_default_hyperparameters().actor_l2
        self.critic_alpha = critic_alpha if critic_alpha else self.get_default_hyperparameters().critic_alpha
        self.critic_l2 = critic_l2 if critic_l2 else self.get_default_hyperparameters().critic_l2
        self.critic_steps = critic_steps if critic_steps else self.get_default_hyperparameters().critic_steps


        self.policy_latest = None
        self.actor_net = actor_net
        self._actor_loss = actor_loss
        # TODO: Optimize with something other than ADAM.
        self.actor_optimizer = torch.optim.Adam(self.actor_net.parameters(), lr=self.actor_alpha, weight_decay=self.actor_l3)

        # Trust that the caller gave a reasonable value network.
        self.critic_net = critic_net
        # Goal of our value network (aka the critic) is to make our actual and expected values be equal.
        self._critic_loss = torch.nn.MSELoss(reduction="mean") # type: ignore
        # TODO: Optimize with something other than ADAM.
        self.critic_optimizer = torch.optim.Adam(self.critic_net.parameters(), lr=self.critic_alpha, weight_decay=self.critic_l2)

    @staticmethod
    def get_default_hyperparameters():
        ret = {}
        ret['actor_loss_mult'] = -1
        ret['actor_alpha'] = .0003
        ret['actor_l2'] = 0.15
        ret['critic_alpha'] = .0001
        ret['critic_l2'] = 0.1
        ret['critic_steps'] = 10
        return types.SimpleNamespace(**ret)

    def recurrent(self):
        return self.actor_net.recurrent() or self.critic_net.recurrent()

    def save_hidden(self):
        ret = {}
        assert self.recurrent()
        if self.actor_net.recurrent(): ret[id(self.actor_net)] = self.actor_net.save_hidden()
        if self.critic_net.recurrent(): ret[id(self.critic_net)] = self.critic_net.save_hidden()
        return ret
            

    def restore_hidden(self, state_dict=None):
        if self.actor_net.recurrent():
            id_actor = id(self.actor_net)
            if state_dict != None and id_actor in state_dict: self.actor_net.restore_hidden(state_dict[id_actor])
            else: self.actor_net.restore_hidden()
        if self.critic_net.recurrent():
            id_critic = id(self.critic_net)
            if state_dict != None and id_critic in state_dict: self.critic_net.restore_hidden(state_dict[id_critic])
            else: self.critic_net.restore_hidden()
            
    def act(self, state):
        return self(state)

    def value(self, state):
        return self.critic_net(state)

    def forward(self, state):
        # Don't return policy information, so as to conform with stochastic agents API.
        actions, logprobs, self.policy_latest = self.actor_net(state)
        return actions, logprobs
        
    def actor_loss(self, task):
        return self._actor_loss(task)

    def critic_loss(self, task):
        # Must reshape states to be a batched 1d array.
        # TODO: Will need different reshaping for CNN's.
        losses = []
        for trajectory in task.trajectories:
            estimated_values = librl.reward._estimate_values(trajectory, self.critic_net).view(-1, 1)
            losses.append(self._critic_loss( estimated_values, trajectory.reward_buffer.view(-1, 1)[:trajectory.done]))
        return sum(losses)

    def steal(self):
        return copy.deepcopy(self.state_dict())
        
    def stuff(self, params):
        self.load_state_dict(params, strict=True)
