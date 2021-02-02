import enum
from more_itertools.more import sample

import numpy.random
import torch

import graphity.graph.generate, graphity.graph.utils

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
        self.state = graphity.lattice.generate.random_glass(self.glass_shape, rng=self.rng)
        return self.state


    def checkpoint(self, *args):
        pass
    def clear_chekpoint(self):
        pass


# Randomly generates adjacency matricies when sampled.
# Doesn't care about checkpointing
class RandomGraphSampler:
    def __init__(self, graph_size=None, seed=None, type=SampleType.Adjacency):
        self.graph_size = graph_size
        self._sample_type = type
        if seed != None:
            self.bit_gen = numpy.random.PCG64(seed)
            self.rng = numpy.random.Generator(self.bit_gen)
        else:
            self.rng = None
    def sample(self, **kwargs):
        if self.sample_type == SampleType.Adjacency:
            self.state = graphity.graph.generate.random_adj_matrix(self.graph_size, rng=self.rng)
        elif self.sample_type == SampleType.Random:
            self.state = graphity.graph.generate.random_graph(self.graph_size, rng=self.rng)
        return self.state
    @property
    def sample_type(self):
        return self._sample_type
    @sample_type.setter
    def sample_type(self, type):
        self._sample_type = type

    def checkpoint(self, *args):
        pass
    def clear_chekpoint(self):
        pass

# Always returns the same adjacency matrix after being sampled once.
# Doesn't care about checkpointing
class FixedGraphSampler:
    def __init__(self, start_state=None, graph_size=None, sample_type = SampleType.Adjacency):
        assert not (graph_size != None and start_state != None)
        if start_state != None:
            if sample_type == SampleType.Adjacency:
                assert graphity.graph.utils.is_adj_matrix(start_state)
            self._start_state = start_state
        elif graph_size != None:
            self.graph_size = graph_size
    @property
    def sample_type(self):
        return self._sample_type
    @sample_type.setter
    def sample_type(self, type):
        self._sample_type = type
        
    def sample(self, **kwargs):
        if not self._start_state: 
            if self._sample_type == SampleType.Adjacency: 
                self._start_state = graphity.graph.generate.random_adj_matrix(self.graph_size)
            elif self._sample_type == SampleType.Random:
                self._start_state = graphity.graph.generate.random_graph(self.graph_size)

        return self._start_state.clone()

    def checkpoint(self, *args):
        pass
    def clear_chekpoint(self):
        pass

class CachedGraphSampler:
    def __init__(self, graph_size=None, seed=None, sample_type=SampleType.Adjacency):
        assert graph_size
        self.sampler = RandomSampler(graph_size, seed, type=sample_type)
        self.sampler.sample_type = sample_type
        self._start_states = {}

    @property
    def sample_type(self):
        return self.sample.sample_type
    @sample_type.setter
    def sample_type(self, type):
        self.sample.sample_type = type

    def sample(self, epoch=None):
        if epoch not in self._start_states: self._start_states[epoch] = self.sampler.sample()
        return self._start_states[epoch].clone()
        
    def reset(self):
        self._start_state = None
    def checkpoint(self, *args):
        pass
    def clear_chekpoint(self):
        pass
# Remembers the best (state, energy) pair passed to checkpoint() unitl the checkpoint is cleared.
# If checkpoint is present, returns the cached value when sampled.
# If there is no checkpoint, will sample from the fallback_sampler passed in init.
class CheckpointGraphSampler:
    def __init__(self, fallback_sampler, sample_type = SampleType.Adjacency):
        self._fallback_sampler = fallback_sampler
        self._fallback_sampler.sample_type = sample_type
        self._state_cache = None

    @property
    def sample_type(self):
        return self._fallback_sampler.sample_type
    @sample_type.setter
    def sample_type(self, type):
        self._fallback_sampler.sample_type = type

    def sample(self, **kwargs):
        if self._state_cache == None:
            return self._fallback_sampler.sample()
        else:
            return self._state_cache[0]

    def checkpoint(self, state, energy):
        #print(energy, self._state_cache[1] if self._state_cache else 0 )
        if not self._state_cache:
            self._state_cache = (state, energy)
        elif self._state_cache[1] > energy:
            #print(f"Accepting state {energy}\n{state} ")
            self._state_cache = (state, energy)

    def clear_chekpoint(self):
        self._state_cache = None