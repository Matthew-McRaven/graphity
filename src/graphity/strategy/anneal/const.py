class ConstBeta:
	def __init__(self, beta):
		self.beta = beta
	def reset(self): pass
	def __call__(self, beta, delta_e):
		return self.beta, 0