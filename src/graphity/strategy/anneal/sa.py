import numpy as np
# Implements Simulated Annealing.
# See: On the Design of an Adaptive Simulated Annealing Algorithm, Cicirello 2009.
#   https://www.cicirello.org/publications/CP2007-Autonomous-Search-Workshop.pdf
class SimulatedAnnealing:
    def __init__(self, alpha, round_length, T0):
        self._timestep = 0
        self.alpha = alpha
        self.T0 = T0
        self.round_length = round_length

    def reset(self):
        self._timestep = 0

    # Implement required pytorch interface
    def __call__(self, beta, delta_e):
        # Cool the system slightly every timestep.
        beta = self.T0 / (self.alpha ** np.floor(self._timestep/self.round_length))
        self._timestep += 1
        return beta, 0