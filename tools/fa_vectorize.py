import io
import itertools
import random
import matplotlib

import matplotlib.pyplot as plt
import networkx as nx
from networkx.algorithms import clique
import numpy as np
from numpy.random import default_rng
import torch.tensor
import torch.nn as nn

import graphity.utils
import graphity.environment.graph.generate as _g
from graphity.environment.graph.learn import *

class NuFACoef(nn.Module):
	def __init__(self, rows, cols):
		"""
		:param graph_size: The number of nodes in an input graph.
		:param rows: The number of rows in the output matrix
		:param cols: The number of columns in the output matrix.
		"""
		super().__init__()
		self.rows, self.cols = rows, cols
		self.coef = nn.parameter.Parameter(torch.zeros((rows, cols)))
		for x in self.parameters():
			if x.dim() > 1: nn.init.kaiming_normal_(x)

	def nas(self, x):
		a_series = [torch.matrix_power(x,i+1) for i in range (1, self.rows+2)]
		a_series = torch.swapaxes(torch.stack(a_series),0,1)
		return a_series

	def oas(self, x):
		a_series = [None for _ in range (self.rows+1)]
		a_series[0] = x.matmul(x)
		for i in range(self.rows): a_series[i+1] = x.matmul(a_series[i])
		return torch.swapaxes(torch.stack(a_series),0,1)

	def nout(self, asrs, x):
		_1 = torch.full(x.shape[-2:], 1.0, dtype=torch.float).to(x.device)
		output = list((_1@asrs[:,i])**(j+1) for i in range(self.rows) for j in range(self.cols))
		output = torch.swapaxes(torch.stack(output), 0,1)
		return output

	def oout(self, asrs, x):
		ret = []
		_1 = torch.full(x.shape[-2:], 1.0, dtype=torch.float).to(x.device)
		for i in range(self.rows):	
			for j in range(self.cols): ret.append((_1 @ (asrs[:,i].float()))**(j+1))
		return torch.swapaxes(torch.stack(ret),0,1)

	def nt(self, pow):
		return torch.diagonal(pow, dim1=-2, dim2=-1).sum(-1)

	def ot(self, pow):
		ret = []
		for x in pow:
			lret = []
			for mat in x:
				lret.append(torch.trace(mat))
			ret.append(torch.stack(lret))
		return torch.stack(ret)

	def nnorm(self, traces):
		norm_pow_mat = torch.stack(list(torch.arange(0, self.cols)+i+2 for i in range(self.rows)))
		numel = traces.shape[-1]*traces.shape[-2]
		numel = torch.full(traces.shape, numel)
		return numel**norm_pow_mat
	def onorm(self, traces):
		ret = []
		for x in traces:
			local = torch.zeros(self.rows, self.cols)
			for (i,j) in itertools.product(range(self.rows), range(self.cols)): local[i,j] = torch.numel(x)**(i+j+2)
			ret.append(local)
		return torch.stack(ret)
	def forward(self, x):
		x = x.float()
		if len(x.shape) == 2: x = x.view(-1, *x.shape)
		elif len(x.shape) > 3: assert 0
		
		a_series = self.nas(x)
		print(a_series)
		oas = self.oas(x)
		assert torch.all(a_series == oas)

		nout, oout = self.nout(a_series,x), self.oout(a_series, x)
		assert torch.all(nout == oout)

		nt, ot = self.nt(nout), self.ot(nout)
		assert(torch.all(nt == ot))

		nt = nt.view(x.shape[0], self.rows, self.cols)
		ot = ot.view(x.shape[0], self.rows, self.cols)
		nnorm, onorm = self.nnorm(nt), self.onorm(ot)
		assert(torch.all(nnorm == onorm))

		assert torch.all(nt/nnorm == ot/onorm)

		return (self.coef * nt/nnorm).sum(dim=[-1,-2])


t = [_g.random_adj_matrix(4) for i in range(1)]
t= torch.stack(t)
n = NuFACoef(2,3)
o = FACoef(2,3)
n.coef = o.coef
print(n.coef), print(o.coef)
print(n(t), o(t))