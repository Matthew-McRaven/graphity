import itertools

import torch
import numpy as np

import graphity.environment.reward, graphity.environment.toggle
import graphity.graph.generate, graphity.graph.utils
"""
A simulator for quantum graphity.
Accepts a hamiltonian H, as well a default graph size.
You may enable or disable self loops.
"""
class Simulator:
    def __init__(self, H=graphity.environment.reward.ASquaredD(2), graph_size=4, allow_self_loop=False, allow_cuda=False):
        self.H = H
        self.graph_size = graph_size
        self.adj_mat = None
        self.allow_cuda = allow_cuda
        self.allow_self_loop = allow_self_loop


    # Reset the environment to a start state---either random or provided.
    # If the supplied adjacency matrix is not the same size as self.graph_size, self.graph_size is updated.
    def reset(self, start_state=None):
        if start_state is not None:
            assert isinstance(start_state, torch.Tensor)
            # Require input matrix t be an adjacency matrix: (all 1's or 0's, square).
            assert graphity.graph.utils.is_adj_matrix(start_state)
            self.state = start_state
            self.graph_size = start_state.shape[-1]
        else:
            # Otherwise depend on our utils facility to give us a good graph.
            self.state = graphity.graph.generate.random_adj_matrix(self.graph_size)
        # TODO: self.state = graphity.hypers.to_cuda(self.state, self.allow_cuda)
        # Simulation-internal state should not provide gradient feedback to system.
        self.state.requires_grad_(False)
        return self.state

    # Apply a list of edge toggles to the current state.
    # Return the state after toggling as well as the reward.
    def step(self, action):
        # Duplicate state so that we have a fresh copy (and we don't destroy replay data)
        next_state = self.state.clone()
        # For each index in the action list, apply the toggles.
        for (i,j) in action:
            graphity.environment.toggle.toggle_edge(int(i), int(j), next_state, self.allow_self_loop)

        # Update self pointer, and score state.
        self.state = next_state
        energy = self.H(self.state)
        #print(energy)
        return self.state, energy

class BatchSimulator:
    def __init__(self, batch_size, H=graphity.environment.reward.ASquaredD(2), graph_size=4, allow_self_loop=False, allow_cuda=False):
        sims = [Simulator(H=H, graph_size=graph_size, allow_self_loop=allow_self_loop, allow_cuda=allow_cuda) for i in range(batch_size)]
        self.sims = np.asarray(sims, dtype=object)
        self.batch_size = batch_size

    def reset(self, start_states=[None]):
        if start_states != [None]:
            assert len(start_states) == self.batch_size
        itsy = itertools.zip_longest(self.sims, start_states)
        bitsy = list(map(lambda s: s[0].reset(s[1]), itsy))
        return torch.stack(bitsy)

    def step(self, actions):

        if len(actions.shape) == 2:
            actions = actions.view(1, *actions.shape)

        state, energy = [], []
        for sim, action in zip(self.sims, actions):
            lstate, lenergy = sim.step(action)
            state.append(lstate)
            energy.append(lenergy)
        return torch.stack(state), torch.stack(energy)