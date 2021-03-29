# Implements Simulated Annealing.
# See: On the Design of an Adaptive Simulated Annealing Algorithm, Cicirello 2009.
#   https://www.cicirello.org/publications/CP2007-Autonomous-Search-Workshop.pdf
@add_agent_attr(allow_callback=True)
class SimulatedAnnealingAgent(MetropolisAgent):
    def __init__(self, sampling_strategy, alpha, round_length, T0):
        super(SimulatedAnnealingAgent, self).__init__(sampling_strategy)
        self.sampling_strategy = sampling_strategy
        self._timestep = 0
        self.alpha = alpha
        self.T0 = T0
        self.round_length = round_length

    def end_epoch(self): self._epoch = self._timestep = 0

    # Implement required pytorch interface
    def forward(self, adj, toggles=1):
        # Cool the system slightly every timestep.
        self.beta = self.T0 / self.alpha ** np.floor(self._timestep/self.round_length)
        self._timestep += 1
        return super().forward(adj, toggles)
        