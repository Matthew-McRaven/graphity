import copy

import torch
import numpy as np
class ShortcutStateMemoizedEpisode:
	"""
	An episode of playback memory for graph evolution with or without metropolis-hasting acceptance.
	All that is required is that actions be indexes to toggle (multiplying by -1) in a given state.
	If this relationship is not true, this class will not work.


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
			# We don't actually store the entire state, we just store the first state and the changes made.
			# This is because the states can be huge (100**2 doubles per t), which the changes are very small
			# (~2 doubles pet t). So, whenever a state is requested, we must rebuild the requested state range.
			if self._field == "state":
				# For slices, we need to generate multiple outputs, all of which require we re-generate states.
				if isinstance(key, slice):
					start, stop, step = key.indices(len(self._array))
					if step < 0: assert False
					elif start > stop: return []
					collected = [idx for idx in range(start, stop, step)]
					
					state = self._array[0].state
					ret = []
					# Play back all actions from the beginning of time until the stopping point.
					for idx, memo in enumerate(self._array[:stop]):
						# Apply change to state if needed.
						if memo.applied: state[tuple(memo.action[0])] *= -1
						# State is mutated in place, so we must create a local copy.
						if idx in collected: ret.append(state.detach().clone())
					return ret

				# Otherwise we just need a single timestep, so play back history (applying updates along the way)
				# until we reach the given key.
				else:
					state = self._array[0].state
					for memo in self._array[:key]:
						if not memo.applied: continue
						state[tuple(memo.action[0])] *= -1  
					# assert (state == self._array[key].state).all()
					return state
			
			# All other fields can be sliced normally, since we maintain full playback history for other fields.
			elif isinstance(key, slice):
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
		super(ShortcutStateMemoizedEpisode, self).__init__()
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
		# Don't log any timesteps other than the first, since these state can be recreated from the actions.
		if t == 0: self.memo[t].state = state
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