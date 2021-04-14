def magnitization(eq_epoch, ending_epoch, sliding_window):
	summed_mag = 0
	for idx, task in enumerate(sliding_window):
		mag = 0
		for buffer in sliding_window[idx][-1]:
			state = buffer.state_buffer[-1]
			#print(state)
			mag = state.float().mean()
			print(mag)

class specific_heat:
	def __init__(self, beta, glass_shape):
		self.beta = beta
		self.glass_shape = glass_shape

	def __call__(self, eq_epoch, ending_epoch, sliding_window):
		num_spins = functools.reduce(lambda prod,item: prod *item, self.glass_shape,1)
		for idx, task in enumerate(sliding_window):
			energies, variances = [], []
			for buffers in task:
				for buffer in buffers:
					energies.extend([buffer.reward_buffer[idx] for idx in range(len(buffer))])	

			batches = [np.random.choice(energies, len(energies)) for i in range(100)]
			
			# Compute per-batch variance
			for batch in batches:
				variance = var(batch)
				variance *= self.beta**2 / num_spins
				variances.append(variance)

			# SQRT(AVG(C^2)-AVG(C)^2)
			# Compute variance of variances, which is specific hea	
			specific_heat = var(energies) * self.beta**2 / num_spins
			error_c = var(variances)**.5
			# See section 3.4 of online book
			print(f"C = {specific_heat} ± {error_c}")