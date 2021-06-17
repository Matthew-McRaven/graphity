import functools
import numpy as np

from .utils import *

def magnitization(trajectories):
	"""
	Compute the average spin (magnitization) of the last element of a trajectory.

	Only operates on the last element because averaging over all elements will create a value close to 0.
	:param trajectories: A list augmented latticies produced by graphity.pipelines.aug
	"""
	mags = []
	for idx, trajectory in enumerate(trajectories):
		# Compute the average spin (aka magnitization). Must cast to float because spins are ints, 
		# and mean() doesn't work on ints.
		mag = trajectory[-1]['state'].float().mean()
		mags.append(mag)
	return mags

class magnetic_susceptibility:
	"""
	Functor to compute the variance of magnitization (aka the magnetic susceptibility).
	"""
	def __init__(self, beta, glass_shape):
		"""
		Initializer for a magnetic susceptibility computation. Stores parameters needed for regularization.

		:param beta: Inverse temperature of the system. Needed to regularize results.
		:param glass_shape: Dimensions of the lattice. Needed to regularize results.
		"""
		self.beta = beta
		self.glass_shape = glass_shape

	def __call__(self, trajectories):
		"""
		Compute the magnetic suceptibilities of a list of trajectories.

		:param trajectories: A list augmented latticies produced by graphity.pipelines.aug
		"""
		num_spins = functools.reduce(lambda prod,item: prod *item, self.glass_shape,1)
		ms = []
		for idx, trajectory in enumerate(trajectories):
			mags = []
			for t in trajectories[idx]:
				# Compute the average spin (aka magnitization). Must cast to float because spins are ints, 
				# and mean() doesn't work on ints.
				mags.append(t['state'].float().mean())	
			# See section 3.3 of online book
			mag_sus = self.beta* num_spins * np.var(mags)
			ms.append(mag_sus)
		return ms

class specific_heat:
	def __init__(self, beta, glass_shape):
		"""
		Initializer for a specific heat computation. Stores parameters needed for regularization.

		:param beta: Inverse temperature of the system. Needed to regularize results.
		:param glass_shape: Dimensions of the lattice. Needed to regularize results.
		"""
		self.beta = beta
		self.glass_shape = glass_shape

	def __call__(self, trajectories):
		"""
		Compute the specific heat of a list of trajectories.

		:param trajectories: A list augmented latticies produced by graphity.pipelines.aug
		"""
		num_spins = functools.reduce(lambda prod,item: prod *item, self.glass_shape,1)
		c = []
		for idx, trajectory in enumerate(trajectories):
			energies, variances = [], []
			for t in trajectories[idx]:
				energies.append(t['energy'])

			# See section 3.4 of online book
			specific_heat = np.var(energies) * self.beta**2 / num_spins
			c.append(specific_heat)
		return c

