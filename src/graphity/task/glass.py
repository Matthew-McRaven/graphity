import torch
from .cc import *
import graphity.replay.any
import graphity.replay.lattice
# Single task object is not shared between episodes within an epoch.
# Otherwise, relay buffers will overwrite each other.
# May re-use tasks acros epochs, so long as you clear_replay() first.
# TODO: Optionally allocate replay buffers on first use.

class GlassTask(ContinuousControlTask):
    def __init__(self, sampler=None, number=None, name=None, **kwargs):
        super(GlassTask, self).__init__(graphity.replay.lattice.ShortcutStateMemoizedEpisode, **kwargs)
        assert name is not None
        self.sampler = sampler
        self.name = name
        self.number = number

    def init_env(self):
        self.env.sampler = self.sampler
class GraphTask(ContinuousControlTask):
    def __init__(self, sampler=None, number=None, name=None, **kwargs):
        super(GraphTask, self).__init__(graphity.replay.any.MemoizedEpisode, **kwargs)
        assert name is not None
        self.sampler = sampler
        self.name = name
        self.number = number

    def init_env(self):
        self.env.sampler = self.sampler