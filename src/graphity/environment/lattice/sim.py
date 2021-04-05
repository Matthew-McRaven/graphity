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
    def reset(self, start=None):
        if start is not None:
            start_state = start
            assert isinstance(start_state, torch.Tensor)
            # Require that new graph is the same size as the environment.
            assert start_state.shape == self.glass_shape
            self.delta_e = None
            self.state = start_state
        else:
            # Otherwise depend on our utils facility to give us a good graph.
            self.state, self.delta_e = random_glass(self.glass_shape), None
        # TODO: self.state = graphity.hypers.to_cuda(self.state, self.allow_cuda)
        # Simulation-internal state should not provide gradient feedback to system.
        self.state = self.state.requires_grad_(False)
        self.contrib = None
        return self.state, self.delta_e

    def evolve(self, sites):
        sites = sites.reshape(-1, len(self.glass_shape))
        # Duplicate state so that we have a fresh copy (and we don't destroy replay data)
        next_state = self.state.clone()
        next_state = next_state.requires_grad_(False)
        # For each index in the action list, apply the toggles.
        changed_sites = []
        for site in sites:
            changed_sites.append((site, next_state[tuple(site)]))
            next_state[tuple(site)] *= -1
        energy, contrib = self.H(next_state, self.contrib, changed_sites)
        return next_state, energy, contrib

    # Apply a list of edge toggles to the current state.
    # Return the state after toggling as well as the reward.
    def step(self, action):
        sites, beta = action
        self.state, self.energy, self.contrib = self.evolve(sites)
        return self.state, 0, self.energy, False, {"contrib":self.contrib}


class RejectionSimulator(SpinGlassSimulator):
    metadata = {}
    def __init__(self, H=IsingHamiltonian(), glass_shape=(4,4), allow_cuda=False):
        super(RejectionSimulator, self).__init__(H, glass_shape, allow_cuda)
        self.rng = Generator(PCG64())
     
    # Apply a list of edge toggles to the current state.
    # Return the state after toggling as well as the reward.
    def step(self, action):
        sites, beta = action
        old_state, old_contribs = self.state, self.contrib
        # Pair of check that ensures that old contribs are not mngled by H.
        #backup = old_contribs.detach().clone() if self.contrib is not None else None
        old_energy, _ = self.H(self.state, self.contrib, [])

        new_state, new_energy, new_contribs = self.evolve(sites)

        delta_e = new_energy - old_energy

        # Perform metropolis-hastings acceptance.
        to_beat = np.exp(-abs(beta*delta_e))
        if delta_e > 0 and self.rng.random() >= to_beat:
            new_state = old_state
            new_contribs = old_contribs
            #assert (new_contribs == backup).all() if backup is not None else True
            new_energy = old_energy

        self.contrib = new_contribs
        self.energy = new_energy
        self.state = new_state
        

        return self.state, delta_e, new_energy, False, {"contrib":self.contrib}

