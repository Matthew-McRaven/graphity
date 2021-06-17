class ConstBeta:
	"""
	A temperature schedule where the temperature remains constant.
	"""
	def __init__(self, beta):
		self.beta = beta
	def reset(self): pass
	def step(self): pass
	def __call__(self, delta_e=None):
		# Log prob is 0 because there is no probabalistic nature to sampling beta.
		return self.beta, 0