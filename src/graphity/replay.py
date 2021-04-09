import copy

import torch
import numpy as np

# Episode where action, state are torch contiguous and simple.
# Therefore, they are torch tensors.
class MemoizedEpisode:
    class Memo:
        def __init__(self):
            self.clear_replay()

        def clear_replay(self):
            self.state = None
            self.action = None
            self.log_prob = None
            self.reward = None
            self.policy = None
        
    def __init__(self, obs_space, act_space, episode_length=200, device='cpu'):
        super(MemoizedEpisode, self).__init__()
        self.done = 0
        self.episode_length = episode_length
        self.memo = np.full([episode_length], None, dtype=object)
        for idx,_ in enumerate(self.memo): self.memo[idx] = self.Memo()
        self.extra = {}
    def __len__(self):
        return self.episode_length
    def __getitem__(self, key):
        return self.memo[key]
    def log_state(self, t, state):
        self[t].state = state
    def log_action(self, t, action, logprob):
        self[t].action = action
        self[t].log_prob = logprob
    def log_rewards(self, t, reward):
        self[t].reward = reward
    def log_policy(self, t, policy):
        self[t].policy = policy
    def log_done(self, t):
        self.done = t

    def log_extra_info(self, t, info_dict):
        assert isinstance(info_dict, dict)

        if t in self.extra: self.extra[t].update(info_dict)
        else: self.extra[t] = info_dict

    def clear_replay(self):
        for memo in self.memo: memo.clear_replay()
        self.extra = {}
        self.done = None