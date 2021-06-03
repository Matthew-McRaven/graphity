import numpy as np
# Implements Simulated Annealing.

class SimulatedAnnealing:
    """
	A cooling schedule where the temperature is decreased gradualually.
    
    It is hard to choose correct alpha, T0, round_length hyperparameters.
    Modern SA algorithms tend to be adaptive, so that we don't have to perform as much hyperparameter tuning.

    See: On the Design of an Adaptive Simulated Annealing Algorithm, Cicirello 2009.
    https://www.cicirello.org/publications/CP2007-Autonomous-Search-Workshop.pdf
	"""
    def __init__(self, alpha, round_length, T0):
        """
        :param alpha: The cooling rate of the system
        :paramm round_length: Number of timesteps between temperature decreases. Can be no less than 1.
        :param T0: Initial temperature. Note, this is regular temperature, not inverse!!
        """
        self._timestep = 0
        self.alpha = alpha
        self.T0 = T0
        self.round_length = round_length

    def reset(self):
        self._timestep = 0

    def __call__(self, beta, delta_e=None):
        """
        Cool the system slightly every timestep.

        Increments the timestep on each invocation.
        """
        beta = self.T0 / (self.alpha ** np.floor(self._timestep/self.round_length))
        self._timestep += 1
        # Log prob is 0 because there is no probabalistic nature to sampling beta.
        return beta, 0