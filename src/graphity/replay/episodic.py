import copy

import torch
import numpy as np

# Enapsulate all replay memory of a single task
# TODO: Allow data to be moved to GPU, and make device required.
class BaseEpisode:
    def __init__(self, obs_space, act_space, episode_length=200, device='cpu', enable_extra = False):
        self.state_buffer =  []
        self.action_buffer = []
        self.logprob_buffer = torch.zeros([episode_length], dtype=torch.float32).to(device) # type: ignore
        self.reward_buffer = torch.zeros([episode_length], dtype=torch.float32).to(device) # type: ignore
        self.policy_buffer = np.full([episode_length], None, dtype=object)
        self.done =  None
        self.enable_extra = enable_extra
        self.extra = {}

    def log_state(self, t, state):
        self.state_buffer[t] = state
    def log_action(self, t, action, logprob):
        self.action_buffer[t] = action
        self.logprob_buffer[t] = logprob
    def log_rewards(self, t, reward):
        assert reward.device == self.reward_buffer.device
        self.reward_buffer[t] = reward
    def log_policy(self, t, policy):
        self.policy_buffer[t]= policy
    def log_done(self, t):
        self.done = t
    def log_extra_info(self, t, info_dict):
        if not self.enable_extra: return
        assert isinstance(info_dict, dict)

        if t in self.extra: self.extra[t].update(info_dict)
        else: self.extra[t] = info_dict

    def clear_replay(self):
        map(lambda x: x.fill_(0).detach_(), [self.logprob_buffer, self.reward_buffer])
        self.policy_buffer.fill(None)
        self.done = None
        self.extra = {}

# Episode where action, state are torch contiguous and simple.
# Therefore, they are torch tensors.
class BoxEpisode(BaseEpisode):
    def __init__(self, obs_space, act_space, episode_length=200, device='cpu', enable_extra = False):
        super(BoxEpisode, self).__init__(obs_space, act_space, episode_length=episode_length,
            device=device, enable_extra=enable_extra)
        self.state_buffer = torch.zeros([episode_length, *obs_space.shape], dtype=librl.utils.convert_np_torch(obs_space.dtype)).to(device) # type: ignore
        self.action_buffer = torch.zeros([episode_length, *act_space.shape], dtype=librl.utils.convert_np_torch(act_space.dtype)).to(device) # type: ignore
    def log_state(self, t, state):
        assert state.device == self.state_buffer.device
        self.state_buffer[t] = state    
    def log_action(self, t, action, logprob):
        assert action.device == self.action_buffer.device
        self.action_buffer[t] = action
        self.logprob_buffer[t] = logprob
    def clear_replay(self):
        super(BoxEpisode, self).clear_replay()
        map(lambda x: x.fill_(0).detach_(), [self.state_buffer, self.action_buffer])

# Episode where action, state are torch contiguous and simple.
# Therefore, they are torch tensors.
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

            
    class View:
        def __init__(self, array, field, shape,):
            self._array = array
            self._field = field
            self.shape = shape
            self.device = array[0].device
        

        def __getitem__(self, key):
            if isinstance(key, slice):
                start, stop, step = key.indices(len(self._array))
                sliced = copy.copy(self)
                sliced._array = self._array[start:stop:step]
                return sliced
            else:
                return getattr(self._array[key], self._field)
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
        if self.enable_extra: self.extra = {}

    def log_state(self, t, state):
        self.memo[t].state = state
    def log_action(self, t, action, logprob):
        self.memo[t].action = action
        self.memo[t].log_prob = logprob
    def log_rewards(self, t, reward):
        self.memo[t].reward = reward
    def log_policy(self, t, policy):
        self.memo[t].policy = policy
    def log_done(self, t):
        self.done = t
    def log_extra_info(self, t, info_dict):
        if not self.enable_extra: return
        assert isinstance(info_dict, dict)

        if t in self.extra: self.extra[t].update(info_dict)
        else: self.extra[t] = info_dict
    def clear_replay(self):
        for memo in self.memo: memo.clear_replay
        if self.enable_extra: self.extra = {}
        self.done = None