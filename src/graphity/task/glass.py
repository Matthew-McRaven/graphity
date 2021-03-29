import torch
from .cc import *
# Single task object is not shared between episodes within an epoch.
# Otherwise, relay buffers will overwrite each other.
# May re-use tasks acros epochs, so long as you clear_replay() first.
# TODO: Optionally allocate replay buffers on first use.

class GlassTask(ContinuousControlTask):
    def __init__(self, sampler=None, number=None, name=None, **kwargs):
        super(GlassTask, self).__init__(**kwargs)
        assert name is not None
        self.sampler = sampler
        self.name = name
        self.number = number

    def init_env(self):
        self.env.sampler = self.sampler