# Placeholder so that package will be built.
import functools

import torch.nn as nn

class FlattenInput(nn.Module):
	def __init__(self, input_dimension):
		super(FlattenInput, self).__init__()
		self.input_dimensions = input_dimension
		self.output_dimension = (functools.reduce(lambda x,y: x*y, input_dimension, 1),)
	def forward(self, input):
		return input.view(-1)