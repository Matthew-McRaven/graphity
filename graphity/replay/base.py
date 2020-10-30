from enum import Enum
import enum

import torch

class LogType(Enum):
    ACTIONS = 1
    REWARDS = 2
    STATES = 3

class BaseReplay:
    def __init__(self):
        self.summarize_epoch = None
        self.clear_each_epoch = False
    def log_actions(self, epoch, episode, t, *args):
        self.log((epoch, episode, t), LogType.ACTIONS, *args)
    def log_rewards(self, epoch, episode, t, *args):
        self.log((epoch, episode, t), LogType.REWARDS, *args)
    def log_state(self, epoch, episode, t, *args):
        self.log((epoch, episode, t), LogType.STATES, *args)
    def log(self, time_tuple, key, *values):
        pass
    def clear(self):
        pass

class DictReplay(BaseReplay):
    def __init__(self):
        super(DictReplay, self).__init__()
        self.clear()

    def log(self, time_tuple, key, *values):
        self._logs[key][time_tuple] = values
    def clear(self):
        self._logs = {x:{} for x in [LogType.ACTIONS,LogType.REWARDS, LogType.STATES]}
    def actions(self):
        return self._logs[LogType.ACTIONS]
    def rewards(self):
        return self._logs[LogType.REWARDS]
    def states(self):
        return self._logs[LogType.STATES]

class BlockReplay(BaseReplay):
    def __init__(self, epochs, episodes_count, episode_length, action_sizes, rewards_sizes, states_sizes):
        super(BlockReplay, self).__init__()
        # Record params about time
        self.epochs = epochs
        self.episodes_count = episodes_count
        self.episode_length = episode_length
        # Record params about size of each element per time.
        self._action_sizes = action_sizes
        self._rewards_sizes = rewards_sizes
        self._states_sizes = states_sizes
        self.clear()

    def log(self, time_tuple, key, *values):
        which = None
        if key == LogType.ACTIONS:
            which = self._actions
        elif key == LogType.REWARDS:
            which = self._rewards
        elif key == LogType.STATES:
            which = self._states
        
        epoch, episode, t = time_tuple
        for i, value in enumerate(values):
            which[i][epoch][episode][t] = value
    def actions(self):
        return self._actions
    def rewards(self):
        return self._rewards
    def states(self):
        return self._states    
    def clear(self):
        self._actions = [torch.zeros(self.epochs, self.episodes_count, self.episode_length, *x) for x in self._action_sizes]
        self._rewards = [torch.zeros(self.epochs, self.episodes_count, self.episode_length, *x) for x in self._rewards_sizes]
        self._states =  [torch.zeros(self.epochs, self.episodes_count, self.episode_length, *x, dtype=torch.uint8) for x in self._states_sizes]