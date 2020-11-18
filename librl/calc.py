import torch

# Given a buffer of actions and policies, compute the log probability of choosing action[t+1] under policy[t]
def old_log_probs(action_buffer, policy_buffer):
	logprob_old = torch.zeros([*policy_buffer.shape], dtype=torch.float32, device=action_buffer.device)
	max_t = policy_buffer.shape[-1] - 1
	# TODO: Figure out what the bootstrap value should be
	logprob_old[max_t] = policy_buffer[max_t].log_prob(action_buffer[max_t]).sum()
	for t in range(max_t):
		logprob_old[t+1] = policy_buffer[t].log_prob(action_buffer[t+1]).sum()
	return logprob_old

# Compute the generalized advantage estimation of our model.
# Caller must compute estimated values outside of this function.
def gae(reward_buffer, estimated_values, gamma, lambd=1):
	advantage = torch.zeros([*reward_buffer.shape], dtype=torch.float32, device=reward_buffer.device)
	max_t = reward_buffer.shape[-1] - 1
	# TODO: Determine how to predict "last" advantage value
	previous, advantage[max_t] = 0, estimated_values[max_t] -  reward_buffer[max_t]
	for t in range(max_t - 1, -1, -1):
		reward = (reward_buffer[t]
				+ gamma*estimated_values[t+1] - estimated_values[t])
		previous = advantage[t] = reward + lambd*gamma*previous
	return advantage

# Compute (in)finite horizion discounted rewards.
# Uses discount factor of gamma, and multiplies lambd(a) by the current element.
# May cause excessively deep gradients, but computes in linear time.	
def discounted_returns(reward_buffer, gamma):
	assert 0.0 < gamma and gamma <= 1.0
	discounted_rewards = torch.zeros([*reward_buffer.shape], dtype=torch.float32, device=reward_buffer.device)
	
	# By iterating backwards over the rewards, we are computing a linear filter on the output.
	# That is: O[t] = I[t] + gamma*O[t+1], where I is the input array and O is the output array.
	# This works by "splitting off" the first term of the summation at t'=t, and then reusing the already-computer sum
	# from t'=t+1..T-1.
	# This approach may accumulate larger roundoff errors when t ≪ stop_time.
	previous = 0
	for t in range(reward_buffer.shape[0]-1, -1, -1):
		previous =  discounted_rewards[t] = reward_buffer[t] + gamma*previous
	return discounted_rewards
