import enum
import itertools

import gym, gym.spaces
import torch
import numpy as np
from numpy.random import Generator, PCG64

from .generate import *
from .reward import ASquaredD

"""
A simulator for quantum spin glasses.
You may choose any hamiltonian designed for spin glasses.
"""
class GraphSimulator(gym.Env):
    metadata = {}
    def __init__(self, H=ASquaredD(2), graph_shape=(4,4), allow_cuda=False):
        self.H = H
        self.graph_shape = graph_shape
        self.allow_cuda = allow_cuda
        # Allow arbitary dimensions for spin glasses
        low = np.array([0 for i in graph_shape])
        hi = np.array([i for i in graph_shape])
        self.action_space = gym.spaces.Box(low=low, high=hi, shape=graph_shape, dtype=np.int8)
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=graph_shape, dtype=np.int8)


    # Reset the environment to a start state---either random or provided.
    # If the supplied adjacency matrix is not the same size as self.glass_shape, self.glass_shape is updated.
    def reset(self, start=None):
        if start is not None:
            assert isinstance(start, torch.Tensor)
            start_state = start.detach().clone()
            # Require that new graph is the same size as the environment.
            assert start_state.shape == self.graph_shape
            self.delta_e = None
            self.state = start_state
        else:
            # Otherwise depend on our utils facility to give us a good graph.
            self.state, self.delta_e = random_adj_matrix(self.graph_shape), None
        # TODO: self.state = graphity.hypers.to_cuda(self.state, self.allow_cuda)
        # Simulation-internal state should not provide gradient feedback to system.
        self.state = self.state.requires_grad_(False)
        # Compute energy and contribs now so that it is immediately available to step() and evolve().
        self.energy, self.contrib = self.H(self.state)
        return self.state, self.delta_e

    def evolve(self, sites):
        sites = sites.reshape(-1, len(self.graph_shape))
        # Duplicate state so that we have a fresh copy (and we don't destroy replay data)
        next_state = self.state.detach().clone()
        #next_state = next_state.requires_grad_(False)
        # For each index in the action list, apply the toggles.
        changed_sites = []
        for site in sites:
            changed_sites.append((site, next_state[tuple(site)]))
            next_state[tuple(site)] = (next_state[tuple(site)] + 1) %2
        energy, contrib = self.H(next_state, self.contrib, changed_sites)
        return next_state, energy, contrib

    # Apply a list of edge toggles to the current state.
    # Return the state after toggling as well as the reward.
    def step(self, action):
        sites, beta = action
        self.state, self.energy, self.contrib = self.evolve(sites)
        return self.state, 0, self.energy, False, {"contrib":self.contrib}