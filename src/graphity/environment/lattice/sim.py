import enum
import itertools

import gym, gym.spaces
import torch
import numpy as np
from numpy.random import Generator, PCG64

from .generate import *
from .reward import IsingHamiltonian

"""
A simulator for quantum spin glasses.
You may choose any hamiltonian designed for spin glasses.
"""
class SpinGlassSimulator(gym.Env):
    metadata = {}
    def __init__(self, H=IsingHamiltonian(), glass_shape=(4,4), allow_cuda=False):
        self.H = H
        self.glass_shape = glass_shape
        self.adj_mat = None
        self.allow_cuda = allow_cuda
        # Allow arbitary dimensions for spin glasses
        low = np.array([*glass_shape, 2])
        hi = np.array([*glass_shape,3])
        self.action_space = gym.spaces.Box(low=low, high=hi, shape=(len(glass_shape)+1,), dtype=np.int8)
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=glass_shape, dtype=np.int8)


    # Reset the environment to a start state---either random or provided.
    # If the supplied adjacency matrix is not the same size as self.glass_shape, self.glass_shape is updated.
    def reset(self, start_state=None):
        if start_state is not None:
            assert isinstance(start_state, torch.Tensor)
            # Require that new graph is the same size as the environment.
            assert start_state.shape == self.glass_shape
            self.state = start_state
        else:
            # Otherwise depend on our utils facility to give us a good graph.
            self.state = graphity.lattice.generate.random_glass(self.glass_shape)
        # TODO: self.state = graphity.hypers.to_cuda(self.state, self.allow_cuda)
        # Simulation-internal state should not provide gradient feedback to system.
        self.state = self.state.requires_grad_(False)
        self.contrib = None
        return self.state

    def evolve(self, sites):
        sites = sites.reshape(-1, len(self.glass_shape))
        # Duplicate state so that we have a fresh copy (and we don't destroy replay data)
        next_state = self.state.clone()
        next_state = next_state.requires_grad_(False)
        # For each index in the action list, apply the toggles.
        changed_sites = []
        for site in sites:
            dims = action
            changed_sites.append((dims, next_state[tuple(dims)]))
            next_state[tuple(dims)] *= -1
        energy, contrib = self.H(next_state, self.contrib, changed_sites)
        return next_state, energy, contrib

    # Apply a list of edge toggles to the current state.
    # Return the state after toggling as well as the reward.
    def step(self, sites, beta):
        self.state, energy, self.contrib = self.evolve(sites)
        return self.state, energy, False, {"contrib":self.contrib}


class RejectionSimulator(SpinGlassSimulator):
    metadata = {}
    def __init__(self, H=IsingHamiltonian(), glass_shape=(4,4), allow_cuda=False):
        super(RejectionSimulator, self).__init__(H, glass_shape, allow_cuda)
        self.rng = Generator(PCG64())
     
    # Apply a list of edge toggles to the current state.
    # Return the state after toggling as well as the reward.
    def step(self, sites, beta):
        old_state, old_contribs = self.state, self.contrib
        old_energy, _ = self.H(self.state, self.contrib, [])

        new_state, new_energy, new_contribs = self.evolve(sites)

        delta_e = new_energy - old_energy

        # Perform metropolis-hastings acceptance.
        to_beat = np.exp(-abs(beta*delta_e))
        if delta_e > 0 and self.rng.random() >= to_beat:
            new_state = old_state
            new_contribs = old_contribs
            new_energy = old_energy

        self.state = new_state
        self.contrib = new_contribs

        return self.state, new_energy, False, {"contrib":self.contrib}

