import gym
import numpy as np
import torch

from .task import Task

import graphity.utils
import graphity.replay
# The current module will be imported before librl.task is finished, so we
# will need to refer to our parent class by relative path.

# Methods to fill a task's sequential replay buffer.
def sample_trajectories(task):
    task.clear_trajectories()
    task.init_env()
    for i in range(task.trajectory_count):
        state, delta_e = task.env.state, task.env.delta_e
        state = graphity.utils.torchize(state, task.device) # type: ignore
        episode = task.replay_ctor(task.env.observation_space, 
            task.env.action_space, task.episode_length, 
            device=task.device
        )
        for t in range(task.episode_length):
            
            episode.log_state(t, state)
            action, logprob_action = task.agent.act(state, delta_e)
            new_state, delta_e, reward, _, extra_info = task.env.step(action)
            episode.log_action(t, action, logprob_action, (new_state == state).all())
            episode.log_extra_info(t, extra_info)

            state = graphity.utils.torchize(new_state, task.device)
            # Don't copy reward in to tensor if it already is one; pytorch gets mad.
            if torch.is_tensor(reward): reward = reward.to(task.device) # type: ignore
            else: reward = torch.tensor(reward).to(task.device) # type: ignore

            episode.log_rewards(t, reward)
            task.agent.end_sweep()

        task.add_trajectory(episode)


class ContinuousControlTask(Task):
    # Aliase batch size as the episode count.
    #episode_count=property(_Task.batch_size, _Task.batch_size)

    # Use sample_trajectories as the default sample, unless otherwise specified.
    def __init__(self, replay_ctor, *, sample_fn = sample_trajectories, 
    env=None, agent=None, trajectories=1, episode_length=100, **kwargs):
        super(ContinuousControlTask,self).__init__(**kwargs)
        assert env is not None and agent is not None
        assert isinstance(env, gym.Env)

        self.env = env
        self.agent = agent
        self._episode_length = episode_length
        self.trajectory_count = trajectories

        self.sample = sample_fn
        self.replay_ctor = replay_ctor
        self.trajectories = []

    # Override in subclass!!
    def init_env(self):
        raise NotImplemented("Please implement this method in your subclass.")

    def add_trajectory(self, trajectory):
        self.trajectories.append(trajectory)
    def clear_trajectories(self):
        self.trajectories = []

    # Let the task control the sampled batch size
    @property
    def episode_length(self):
        return self._episode_length
    @episode_length.setter
    def episode_length(self, value):
        self._episode_length = value
