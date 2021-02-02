import enum
import itertools

import gym, gym.spaces
import torch
import numpy as np

import graphity.environment.biqg.reward
import graphity.graph.generate, graphity.graph.utils
from .reward import *

class GraphSimulator(gym.Env):
    metadata = {}
    def __init__(self, H=ASquaredD(2), graph_size=4, allow_self_loop=False, allow_cuda=False):
        self.H = H
        self.graph_size = graph_size
        self.adj_mat = None
        self.allow_cuda = allow_cuda
        self.allow_self_loop = allow_self_loop
        self.action_space = gym.spaces.Box(low=0, high=graph_size-1, shape=(1,2), dtype=np.int8)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(graph_size, graph_size), dtype=np.int8)


    # Reset the environment to a start state---either random or provided.
    # If the supplied adjacency matrix is not the same size as self.graph_size, self.graph_size is updated.
    def reset(self, start_state=None):
        if start_state is not None:
            assert isinstance(start_state, torch.Tensor)
            # Require that new graph is the same size as the environment.
            assert start_state.shape == (self.graph_size,self.graph_size)
            # Require input matrix t be an adjacency matrix: (all 1's or 0's, square).
            assert graphity.graph.utils.is_adj_matrix(start_state)
            self.state = start_state
        else:
            # Otherwise depend on our utils facility to give us a good graph.
            self.state = graphity.graph.generate.random_adj_matrix(self.graph_size)
        # TODO: self.state = graphity.hypers.to_cuda(self.state, self.allow_cuda)
        # Simulation-internal state should not provide gradient feedback to system.
        self.state = self.state.requires_grad_(False)
        self.contrib = None
        return self.state
        
    # Apply a list of edge toggles to the current state.
    # Return the state after toggling as well as the reward.
    def step(self, actions):
        actions = actions.reshape(-1, 3)
        # Duplicate state so that we have a fresh copy (and we don't destroy replay data)
        self.state = self.state.clone().requires_grad_(False)
        # For each index in the action list, apply the toggles.
        changed_sites = []
        for (i,j, _) in actions:
            # If our action falls on the diagonal, only allow change if we allow self loops.
            if i == j and self.allow_self_loop:
                changed_sites.append((i, j, self.state[j, i]))
                self.state[i, j] = self.state[i, j] ^ 1
            # Otherwise, toggle undirected edge between both nodes.
            else: 
                if True: 
                    changed_sites.append((j, i, self.state[j, i]))
                    self.state[j, i] ^= 1
                changed_sites.append((i, j, self.state[i, j]))
                self.state[i, j] = self.state[i, j]^1

        # Update self pointer, and score state.
        energy, self.contrib = self.H(self.state, self.contrib, changed_sites)
        return self.state, energy, False, {"contrib": self.contrib}