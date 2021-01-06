import numpy.random

import graphity.graph.generate, graphity.graph.utils

# Randomly generates adjacency matricies when sampled.
# Doesn't care about checkpointing
class RandomSampler:
    def __init__(self, graph_size=None, seed=None):
        self.graph_size = graph_size
        if seed != None:
            self.bit_gen = numpy.random.PCG64(seed)
            self.rng = numpy.random.Generator(self.bit_gen)
        else:
            self.rng = None
    def sample(self):
        self.state = graphity.graph.generate.random_adj_matrix(self.graph_size, rng=self.rng)
        return self.state
    
    def checkpoint(self, *args):
        pass
    def clear_chekpoint(self):
        pass

# Always returns the same adjacency matrix after being sampled once.
# Doesn't care about checkpointing
class FixedSampler:
    def __init__(self, start_state=None, graph_size=None):
        assert not (graph_size != None and start_state != None)
        if start_state != None:
            assert graphity.graph.utils.is_adj_matrix(start_state)
            self._start_state = start_state
        elif graph_size != None:
            self._start_state = graphity.graph.generate.random_adj_matrix(graph_size)

    def sample(self):
        return self._start_state.clone()

    def checkpoint(self, *args):
        pass
    def clear_chekpoint(self):
        pass
class CachedSampler:
    def __init__(self, graph_size=None, seed=None):
        assert graph_size
        self.sampler = RandomSampler(graph_size, seed)
        self._start_state = None

    def sample(self):
        if self._start_state == None: self._start_state = self.sampler.sample()
        return self._start_state.clone()
    def reset(self):
        self._start_state = None
    def checkpoint(self, *args):
        pass
    def clear_chekpoint(self):
        pass
# Remembers the best (state, energy) pair passed to checkpoint() unitl the checkpoint is cleared.
# If checkpoint is present, returns the cached value when sampled.
# If there is no checkpoint, will sample from the fallback_sampler passed in init.
class CheckpointSampler:
    def __init__(self, fallback_sampler):
        self._fallback_sampler = fallback_sampler
        self._state_cache = None

    def sample(self):
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