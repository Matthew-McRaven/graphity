import functools
import numpy as np

from .utils import *

def magnitization(trajectories):
	mags = []
	for idx, trajectory in enumerate(trajectories):
		mag = trajectory[-1]['state'].float().mean()
		mags.extend([mag])
	return mags

class specific_heat:
	def __init__(self, beta, glass_shape):
		self.beta = beta
		self.glass_shape = glass_shape

	def __call__(self, trajectories):
		num_spins = functools.reduce(lambda prod,item: prod *item, self.glass_shape,1)
		c = []
		for idx, trajectory in enumerate(trajectories):
			energies, variances = [], []
			for t in trajectories[idx]:
				energies.append(t['energy'])	

			batches = [np.random.choice(energies, len(energies)) for i in range(100)]
			
			# Compute per-batch variance
			for batch in batches:
				variance = var(batch)
				variance *= self.beta**2 / num_spins
				variances.append(variance)

			# SQRT(AVG(C^2)-AVG(C)^2)
			# Compute variance of variances, which is specific hea	
			specific_heat = var(energies) * self.beta**2 / num_spins
			c.append(specific_heat)
			error_c = var(variances)**.5
			# See section 3.4 of online book
			#print(f"C = {specific_heat} ± {error_c}")
		return c