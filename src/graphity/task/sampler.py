import enum
from more_itertools.more import sample

import numpy.random
import torch
import graphity.environment.lattice

class SampleType(enum.Enum):
    Adjacency = enum.auto()
    Random = enum.auto()

# Randomly generates adjacency matricies when sampled.
# Doesn't care about checkpointing
class RandomGlassSampler:
    def __init__(self, glass_shape=None, seed=None):
        self.glass_shape = glass_shape
        self._sample_type = type
        if seed != None:
            self.bit_gen = numpy.random.PCG64(seed)
            self.rng = numpy.random.Generator(self.bit_gen)
        else:
            self.rng = None
    def sample(self, **kwargs):
        self.state = graphity.environment.lattice.random_glass(self.glass_shape, rng=self.rng)
        return self.state


    def checkpoint(self, *args):
        pass
    def clear_chekpoint(self):
        pass