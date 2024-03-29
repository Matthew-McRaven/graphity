import enum
import itertools

import gym, gym.spaces
import torch
import numpy as np
from numpy.random import Generator, PCG64

from .generate import *


class GraphSimulator(gym.Env):
    """
    A simulator for spacetime graphs which unconditionally accepts edge toggles.
    You may choose any hamiltonian designed for spacetime graphs.
    """
    def __init__(self, H, graph_shape=(4,4), allow_cuda=False):
        """
        :param H: A Hamiltonian designed for spacetime graphs.
        :param graph_shape: The dimension of graph to be used in the simulator.
        :param allow_cuda: Can we do computations on the GPU? Currently ignored (2021-06-03).
        """
        self.H = H
        self.graph_shape = graph_shape
        self.allow_cuda = allow_cuda

        # Adjacency mats only make sense as 2 dim objects.
        assert len(graph_shape) == 2
        low = np.array([0 for i in graph_shape])
        hi = np.array([i for i in graph_shape])
        self.action_space = gym.spaces.Box(low=low, high=hi, shape=(len(graph_shape),), dtype=np.int8)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=graph_shape, dtype=np.int8)

    def reset(self, start=None):
        """
        Reset the environment to a start state---either random or provided.
        If the supplied graph is not the same size as self.glass_shape, self.glass_shape is updated.

        :param start: A graph from which simulation begins. If None, a random start state will be generated.
        """
        if start is not None:
            assert isinstance(start, torch.Tensor)
            start_state = start.detach().clone()
            # Require that new graph is the same size as the environment.
            assert start_state.shape == self.graph_shape
            self.delta_e = None
            self.state = start_state
        else:
            # Otherwise depend on our utils facility to give us a good graph.
            self.state, self.delta_e = random_adj_matrix(self.graph_shape[0]), None
        # TODO: self.state = graphity.hypers.to_cuda(self.state, self.allow_cuda)
        # Simulation-internal state should not provide gradient feedback to system.
        self.state = self.state.requires_grad_(False)
        # Compute energy and contribs now so that it is immediately available to step() and evolve().
        self.energy, self.contrib = self.H(self.state)
        return self.state, self.delta_e

    def evolve(self, sites):
        """
        Time evolve the current state by applying the list of sites contained in sites.

        The elements of site should have the same len as self.glass_shape.
        :param sites: A list of tuples. Each tuple contains a site to be toggled.
        """
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

    def step(self, action):
        """
        Modify the current state by applying the action to the current state.

        :param action: A tuple of sites to change and the beta at which the simulation is to be run.
        """
        sites, beta = action
        self.state, self.energy, self.contrib = self.evolve(sites)
        return self.state, 0, self.energy, False, {"contrib":self.contrib}

class RejectionSimulator(GraphSimulator):
    """
    A simulator for spacetime graphs which conditionally accepts spin flips according to metropolis-hastings.
    You may choose any hamiltonian designed for spacetime graphs.
    """
    def __init__(self, H, graph_shape=(4,4), allow_cuda=False):
        """
        :param H: A Hamiltonian designed for spacetime graphs.
        :param graph_shape: The dimension of spacetime grpahs to be used in the simulator.
        :param allow_cuda: Can we do computations on the GPU? Currently ignored (2021-06-03).
        """

        super(RejectionSimulator, self).__init__(H, graph_shape, allow_cuda)
        self.rng = Generator(PCG64())
     
    def step(self, action):
        """
        Modify the current state by applying the action to the current state.
        If the action is disadvantageous, it will be conditionally rejected according to the metropolis-hastings algorithm.

        :param action: A tuple of sites to change and the beta at which the simulation is to be run.
        """
        sites, beta = action
        old_state, old_contribs, old_energy = self.state, self.contrib, self.energy
        # Pair of check that ensures that old contribs are not mngled by H.
        #backup = old_contribs.detach().clone() if self.contrib is not None else None
        #old_energy, _ = self.H(self.state, self.contrib, [])

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