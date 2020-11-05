from enum import Enum
import enum

import torch
import numpy as np

from graphity.hypers import to_cuda

# Replay buffer that records past states.
# Must be cleared between epochs, or after each optimizer "step()".
# Logged elements are kept on whichever device they were allocated on.
class StateBuffer:
    def __init__(self, episode_count, episode_len, state_size, allow_cuda=False):
        self.episode_count = episode_count
        self.episode_len = episode_len
        self.states = to_cuda(torch.zeros([episode_count, episode_len, *state_size], dtype=torch.uint8), allow_cuda)

    def log_state(self, episode, t, state):
        self.states[episode][t] = state

    def clear(self):
        # Must detach gradients from old training loops, otherwise subsequent
        # optimizer steps will crash from missing grads.
        # Use "_" methods because they are in place, and won't wastefully re-allocate memory.
        self.states.fill_(0)
        self.states.detach_()

# Replay buffer that records past actions and action probabilities.
# Must be cleared between epochs, or after each optimizer "step()".
# Logged elements are kept on whichever device they were allocated on.
class ActionBuffer:
    def __init__(self, episode_count, episode_len, action_size, allow_cuda=False):
        self.episode_count = episode_count
        self.episode_len = episode_len
        self.actions = to_cuda(torch.zeros([episode_count, episode_len, *action_size], dtype=torch.int16), allow_cuda)
        self.logprob_actions = to_cuda(torch.zeros([episode_count, episode_len, 1], dtype=torch.float32), allow_cuda)

    def log_action(self, episode, t, action, logprob=0):
        self.actions[episode][t] = action
        self.logprob_actions[episode][t] = logprob

    def clear(self):
        # Must detach gradients from old training loops, otherwise subsequent
        # optimizer steps will crash from missing grads.
        # Use "_" methods because they are in place, and won't wastefully re-allocate memory.
        self.actions.fill_(0)
        self.logprob_actions.fill_(0)
        self.actions.detach_()
        self.logprob_actions.detach_()

# Replay buffer that records past rewards.
# Must be cleared between epochs, or after each optimizer "step()".
# Logged elements are kept on whichever device they were allocated on.
# Does not implicitly accumulate discounted rewards.
# For discounted returns, see graphity.replay.ReturnAccumulator.
class RewardBuffer:
    def __init__(self, episode_count, episode_len, reward_size, allow_cuda=False):
        self.episode_count = episode_count
        self.episode_len = episode_len
        self.rewards = to_cuda(torch.zeros([episode_count, episode_len, *reward_size], dtype=torch.float32), allow_cuda)

    def log_rewards(self, episode, t, reward):
        self.rewards[episode][t] = reward

    def clear(self):
        # Must detach gradients from old training loops, otherwise subsequent
        # optimizer steps will crash from missing grads.
        # Use "_" methods because they are in place, and won't wastefully re-allocate memory.
        self.rewards.fill_(0)
        self.rewards.detach_()

# Replay buffer that records past policy objects.
# Should be cleared between epochs, for consistency with other replay buffers.
# However, it will not crash if you forget to clear.
# It stores pointers to python policy objects, which are described elsewhere.
class PolicyBuffer:
    def __init__(self, episode_count, episode_len):
        self.episode_count = episode_count
        self.episode_len = episode_len
        self.policies = np.zeros([episode_count, episode_len], dtype=object)

    def log_policy(self, episode, t, policy):
        self.policies[episode][t] = policy

    def clear(self):
        self.policies.fill(0)

# Compute (in)finite horizion discounted rewards.
# Uses discount factor of gamma, and multiplies lambd(a) by the current element.
# May cause excessively deep gradients, but computes in linear time.
class ReturnAccumulator:
    def __init__(self, rewards, gamma, lambd=-1):
        assert 0.0 < gamma and gamma <= 1.0
        self.gamma = gamma
        self.lambd = lambd
        self.discounted_rewards = torch.zeros([rewards.episode_count, rewards.episode_len, 1], dtype=torch.float32,
                                              device=rewards.rewards.device)
        previous = 0
        # By iterating backwards over the rewards, we are computing a linear filter on the output.
        # That is: O[t] = I[t] + gamma*O[t+1], where I is the input array and O is the output array.
        # This works by "splitting off" the first term of the summation at t'=t, and then reusing the already-computer sum
        # from t'=t+1..T-1.
        # This approach may accumulate larger roundoff errors when t ≪ stop_time.
        for episode in range(rewards.episode_count):
            for t in range(rewards.episode_len-1, -1, -1):
                previous = self.discounted_rewards[episode][t] = lambd*rewards.rewards[episode][t] + gamma*previous
