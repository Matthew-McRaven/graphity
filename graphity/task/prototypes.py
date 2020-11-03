import torch
import graphity.environment.toggle
import graphity.replay

# Describes a family of related tasks
class TaskDefinition:
    def __init__(self, sampler, loss, hypers):
        self.sampler = sampler
        self.loss = loss
        self.hypers = hypers
    def instance(self):
        return Task(self)

# Single task object is not shared between episodes within an epoch.
# Otherwise, relay buffers will overwrite each other.
# May re-use tasks acros epochs, so long as you clear_replay() first.
# TODO: Optionally allocate replay buffers on first use.
class Task:
    def __init__(self, definition):
        self.sampler = definition.sampler
        self.loss = definition.loss
        self.hypers = definition.hypers
        self.state_buffer = graphity.replay.StateBuffer(1, self.hypers['episode_length'], (self.hypers['graph_size'], self.hypers['graph_size']))
        self.action_buffer = graphity.replay.ActionBuffer(1, self.hypers['episode_length'], (self.hypers['toggles_per_step'], 2))
        self.reward_buffer = graphity.replay.RewardBuffer(1, self.hypers['episode_length'], (1,))
        self.policy_buffer = graphity.replay.PolicyBuffer(1, self.hypers['episode_length'])

    def sample_starting_point(self):
        return self.sampler.sample()

    # Forward calls for logging on to proper replay buffers.
    def log_state(self, t, state):
        self.state_buffer.log_state(0, t, state)
    def log_action(self, t, action, logprob):
        self.action_buffer.log_action(0, t, action, logprob)
    def log_rewards(self, t, reward):
        self.reward_buffer.log_rewards(0, t, reward)
    def log_policy(self, t, policy):
        self.policy_buffer.log_policy(0, t, policy)

    # Must clear each of the replay buffers, or re-using tasks between epochs will crash.
    def clear_replay(self):
        [x.clear() for x in [self.state_buffer, self.reward_buffer, self.action_buffer, self.policy_buffer]]

    # Find the minimum energy state and return the (state, energy, index) tuple.
    def min(self):
        # Find the state with the smallest energy
        mindex = torch.argmin(self.reward_buffer.rewards)
        # Compute the episode, time of the minimum energy
        episode, t = mindex // self.reward_buffer.episode_len, mindex % self.reward_buffer.episode_len

        graph = self.state_buffer.states[episode][t]
        for (i,j) in self.action_buffer.actions[episode][t]:
            graphity.environment.toggle.toggle_edge(int(i), int(j), graph, False)
        
        return graph, self.reward_buffer.rewards[episode][t], t