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