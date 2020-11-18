import torch
import graphity.environment.toggle
import graphity.replay

# Describes a family of related tasks
# Single task object is not shared between episodes within an epoch.
# Otherwise, replay buffers will overwrite each other.
class Task():
    class Definition:
        def __init__(self, ctor, **kwargs):
            self.task_ctor = ctor
            self.task_kwargs = kwargs
        def instance(self):
            return self.task_ctor(**self.task_kwargs)
    def __init__(self, sample_fn = None, env=None, agent=None, trajectories=1):
        assert env is not None and agent is not None and sample_fn is not None
        self.env = env
        self.agent = agent
        self.trajectory_count = trajectories
        self.sample = sample_fn
        self.trajectories = []
    # Override in subclass!!
    def init_env(self):
        pass

    def add_trajectory(self, trajectory):
        self.trajectories.append(trajectory)
    def clear_trajectories(self):
        self.trajectories = []

# Single task object is not shared between episodes within an epoch.
# Otherwise, relay buffers will overwrite each other.
# May re-use tasks acros epochs, so long as you clear_replay() first.
# TODO: Optionally allocate replay buffers on first use.
class GraphTask(Task):
    def __init__(self, sampler=None, loss=None, **kwargs):
        super(GraphTask, self).__init__(**kwargs)
        assert loss is not None

        self.sampler = sampler
        self.loss = loss

    def init_env(self):
        self.env.sampler = self.sampler



    # Find the minimum energy state and return the (state, energy, index) tuple.
def min(graphtask):
    # Find the state with the smallest energy
    mindex = torch.argmin(self.reward_buffer.rewards)
    # Compute the episode, time of the minimum energy
    episode, t = mindex // self.reward_buffer.episode_len, mindex % self.reward_buffer.episode_len

    graph = self.state_buffer.states[episode][t]
    for (i,j) in self.action_buffer.actions[episode][t]:
        graphity.environment.toggle.toggle_edge(int(i), int(j), graph, False)
    
    return graph, self.reward_buffer.rewards[episode][t], t