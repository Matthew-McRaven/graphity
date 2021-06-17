import enum
from more_itertools.more import sample

import numpy.random
import torch

import graphity.environment.graph
import graphity.environment.lattice


# Randomly generates lattices when sampled.
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

# Randomly generates adjacency matricies when sampled.
class RandomGraphSampler:
    def __init__(self, adj_shape=None, seed=None):
        self.adj_shape = glass_shape
        self._sample_type = adj_shape
        if seed != None:
            self.bit_gen = numpy.random.PCG64(seed)
            self.rng = numpy.random.Generator(self.bit_gen)
        else:
            self.rng = None
            
    def sample(self, **kwargs):
        self.state = graphity.environment.graph.random_adj_matrix(self.adj_shape, rng=self.rng)
        return self.state