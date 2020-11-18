
import torch
import numpy as np

import librl.utils

# Enapsulate all replay memory of a single task
# TODO: Allow data to be moved to GPU.
class Episode:
	def __init__(self, obs_space, act_space, episode_length=200, allow_cuda=False):
		self.state_buffer = torch.zeros([episode_length, *obs_space.shape], dtype=librl.utils.convert_np_torch(obs_space.dtype))
		self.action_buffer = torch.zeros([episode_length, *act_space.shape], dtype=librl.utils.convert_np_torch(act_space.dtype))
		self.logprob_buffer = torch.zeros([episode_length], dtype=torch.float32)
		self.reward_buffer = torch.zeros([episode_length], dtype=torch.float32)
		self.policy_buffer = np.full([episode_length], None, dtype=object)
		self.done =  None

	def log_state(self, t, state):
		self.state_buffer[t] = state
	def log_action(self, t, action, logprob):
		self.action_buffer[t] = action
		self.logprob_buffer[t] = logprob
	def log_rewards(self, t, reward):
		self.reward_buffer[t] = reward
	def log_policy(self, t, policy):
		self.policy_buffer[t]= policy
	def log_done(self, t):
		self.done = t

	def clear_replay(self):
		map(lambda x: x.fill_(0).detach_(), [self.state_buffer, self.action_buffer, self.logprob_buffer, self.reward_buffer])
		self.policy_buffer.fill(None)
		self.done = None