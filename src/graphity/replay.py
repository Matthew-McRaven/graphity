import copy

import torch
import numpy as np
class MemoizedEpisode:
    class Memo:
        def __init__(self, device='cpu'):
            self.clear_replay()
            self.device = device

        def clear_replay(self):
            self.state = None
            self.action = None
            self.log_prob = None
            self.reward = None
            self.policy = None
            self.applied = False

            
    class View:
        def __init__(self, array, field, shape,):
            self._array = array
            self._field = field
            self.shape = shape
            self.device = array[0].device
        

        def __getitem__(self, key):
            if self._field == "state":
                if isinstance(key, slice):
                    assert False
                else:
                    state = self._array[0].state
                    for memo in self._array[0:key]:
                        if not memo.applied: continue
                        state[tuple(memo.action[0])] *= -1  
                   # assert (state == self._array[key].state).all()
                    return state
            elif isinstance(key, slice):
                start, stop, step = key.indices(len(self._array))
                sliced = copy.copy(self)
                sliced._array = self._array[start:stop:step]
                return sliced
            
            else: return getattr(self._array[key], self._field)
        def __len__(self):
            return len(self._array)
        
    def __init__(self, obs_space, act_space, episode_length=200, device='cpu', enable_extra = False):
        super(MemoizedEpisode, self).__init__()
        self.done = 0
        self.enable_extra = enable_extra
        self.memo = np.full([episode_length], None, dtype=object)
        for idx,_ in enumerate(self.memo): self.memo[idx] = self.Memo()
        self.state_buffer = self.View(self.memo, 'state', obs_space.shape)
        self.action_buffer = self.View(self.memo, 'action', act_space.shape)
        self.logprob_buffer = self.View(self.memo, 'log_prob', (episode_length,))
        self.reward_buffer = self.View(self.memo, 'reward', (episode_length,))
        self.policy_buffer = self.View(self.memo, 'policy', (episode_length,))
        self.extra = {}
    def __len__(self):
        return len(self.memo)

    def log_state(self, t, state):
        if t == 0: 
            self.memo[t].state = state
    def log_action(self, t, action, logprob, rejected):
        self.memo[t].action = action
        self.memo[t].log_prob = logprob
        self.memo[t].applied = not rejected
    def log_rewards(self, t, reward):
        self.memo[t].reward = reward
    def log_policy(self, t, policy):
        self.memo[t].policy = policy
    def log_done(self, t):
        self.done = t
    def log_extra_info(self, t, info_dict):
        assert isinstance(info_dict, dict)

        if t in self.extra: self.extra[t].update(info_dict)
        else: self.extra[t] = info_dict

    def clear_replay(self):
        for memo in self.memo: memo.clear_replay
        self.extra = {}
        self.done = None