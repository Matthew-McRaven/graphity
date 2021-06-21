import copy

import torch
import numpy as np
class MemoizedEpisode:
	"""
	An episode of playback memory for graph/lattice evolution with or without metropolis-hasting acceptance.
	No special relationship between state:action pairs is required.	
	"""
	class Memo:
		"""
		A Memo stores all relevant information about an agnet-environment interaction for a single timestep.
		"""
		def __init__(self, device='cpu'):
			"""
			:param device: A torch device on which the Memo should store its tensors.
			"""
			self.clear_replay()
			self.device = device

		def clear_replay(self):
			"""
			Reset the memory of this memo to an unitialized state.
			"""
			# The state presented to the agent before making an action decision.
			self.state = None
			# The action submitted to the environment.
			self.action = None
			# The reward that came from the environment.
			self.reward = None
			# Did the environment actually apply the move.
			self.applied = False

			
	class View:
		"""
		A view re-indexes Memos to give the appearance that fields (like Memo.reward) are stored in a contiguous array
		rather than inside individual Memo objects.
		This convience class allows for complex slicing and indexing operations that would otherwise be impossible.

		"""
		def __init__(self, array, field, shape,):
			"""
			:param _array: An array of Memos on which this view operates.
			:param _field: A string name of a
			"""
			self._array = array
			self._field = field
			self.shape = shape
			self.device = array[0].device
		

		def __getitem__(self, key):
			if isinstance(key, slice):
				# Unpack slice to its components.
				start, stop, step = key.indices(len(self._array))
				# In order to implement slicing, we will duplicate the current view, with its array replaced
				# by a copy of the array with slicing applied.
				sliced = copy.copy(self)
				sliced._array = self._array[start:stop:step]
				return sliced

			# Otherwise just return the individual item being asked for.
			else: return getattr(self._array[key], self._field)

		def __len__(self):
			return len(self._array)
		
	def __init__(self, obs_space, act_space, episode_length=200, device='cpu'):
		"""
		:param obs_space: A gym space object whose shape is that of the state tensor.
		:param act_space: A gym space object whose shape is that of the state action.
		:param episode_length: The maximum number of observations that can be stored in the replay buffer.
		:param device: A torch device on which to store local tensors. ignored as of 20210621.
		"""
		super(MemoizedEpisode, self).__init__()
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
		"""
		Log the state of the system at timestep t.
		:param t: The timestep in question.
		:param state: A tensor describing the state of the system.
		"""
		self.memo[t].state = state
	def log_action(self, t, action, logprob, rejected):
		"""
		Log the action of the agent at timestep t.
		:param t: The timestep in question.
		:param action: A site to be toggled.
		:param logprob: The probability of choosing that action.
		:param rejected: Was the action rejected by metropolis-hastings acceptance?
		"""
		self.memo[t].action = action
		self.memo[t].log_prob = logprob
		self.memo[t].applied = not rejected

	def log_rewards(self, t, reward):
		"""
		Log the energy of the system at timestep t after applying an action to the current state.
		:param t: The timestep in question.
		:param reward: A tensor describing the energy of the system after applying an action to the current state.
		"""
		self.memo[t].reward = reward


	def log_extra_info(self, t, info_dict):
		"""
		Log additional information for timestep t.
		:param t: The timestep in question.
		:param reward: A dict containing extra information to be stored at timestep t.
		"""
		assert isinstance(info_dict, dict)

		if t in self.extra: self.extra[t].update(info_dict)
		else: self.extra[t] = info_dict

	def clear_replay(self):
		"""
		Clear all replay memory.
		"""
		for memo in self.memo: memo.clear_replay()
		self.extra = {}