import functools
import math

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import torchvision
import torch.nn as nn
from sklearn.model_selection import KFold
from sklearn import datasets, svm, metrics, decomposition, neighbors, linear_model
from sklearn.model_selection import train_test_split

import graphity.data
import graphity.environment.graph
from graphity.environment.graph.generate import random_adj_matrix, random_pure_graph
import graphity.read_data
import graphity.utils

def conv_output_shape(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
	"""
	Given the parameters of a 2D convolution, compute the height and width of the output.
	"""
	if type(kernel_size) is not tuple:
		kernel_size = (kernel_size, kernel_size)
	h = math.floor( ((h_w[0] + (2 * pad) - ( dilation * (kernel_size[0] - 1) ) - 1 )/ stride) + 1)
	w = math.floor( ((h_w[1] + (2 * pad) - ( dilation * (kernel_size[1] - 1) ) - 1 )/ stride) + 1)
	return h, w

class Raw(nn.Module):
	"""
	Learn a Hamiltonian term that is a linear function of the input graph.
	"""
	def __init__(self, graph_size):
		"""
		:param graph_size: The number of nodes in an input graph.
		"""
		super().__init__()
		self.graph_size = graph_size
		self._M = nn.parameter.Parameter(torch.zeros((1, graph_size, graph_size)))
		for x in self.parameters():
			if x.dim() > 1: nn.init.kaiming_normal_(x)

	def apply_coef(self, x):
		"""
		Apply this class's transformation to a single 2D input.
		"""
		return self._M @ x

	def stuff_weights(self, weights):
		self._M.weight = weights
	def weight_count(self):
		return functools.reduce(lambda x, y: x*y, self._M.shape, 1)

	def forward(self, x):
		x = x.view(-1, self.graph_size, self.graph_size)
		return torch.stack([self.apply_coef(i) for i in x[:]]).to(x.device)

class Conv(nn.Module):
	def __init__(self, graph_size, rows, cols, channels=1, k=6):
		"""
		:param graph_size: The number of nodes in an input graph.
		:param rows: The number of rows in the output matrix
		:param cols: The number of columns in the output matrix.
		:param channels: The number of channels in the convolutional layer.
		:param k: The kernel size of the convolutional layer.
		"""
		super().__init__()
		self.graph_size = graph_size
		self.rows, self.cols = rows, cols
		self.coef = nn.parameter.Parameter(torch.zeros((rows, cols)))
		self.conv = torch.nn.Conv2d(1, channels, (k,k))
		hw = conv_output_shape((graph_size, graph_size), (k,k))
		self.lin = torch.nn.Linear(hw[0]*hw[1]*channels, rows*cols)
		for x in self.parameters():
			if x.dim() > 1: nn.init.kaiming_normal_(x)

	def apply_coef(self, x):
		"""
		Apply this class's transformation to a single 2D input.
		"""
		x = x.view(1,1, *x.shape)
		x = self.conv(x)
		x = self.lin(x.view(-1)).view(self.rows, self.cols)
		return (self.coef * x).sum()

	def stuff_weights(self, weights): assert(0)
	def weight_count(self): assert(0)

	def forward(self, x):
		x = x.view(-1, self.graph_size, self.graph_size)
		return torch.stack([self.apply_coef(i) for i in x[:]]).to(x.device)

class ConvTrace(nn.Module):
	def __init__(self, rows, cols, channels=5, k=6):
		"""
		:param graph_size: The number of nodes in an input graph.
		:param rows: The number of rows in the output matrix
		:param cols: The number of columns in the output matrix.
		:param channels: The number of channels in the convolutional layer.
		:param k: The kernel size of the convolutional layer.
		"""
		super().__init__()
		self.rows, self.cols, self.channels = rows, cols, channels
		self.coef = nn.parameter.Parameter(torch.zeros((channels, rows, cols)))
		self.conv = torch.nn.Conv2d(1, channels, (k,k))

		for x in self.parameters():
			if x.dim() > 1: nn.init.kaiming_normal_(x)

	def apply_coef(self, x):
		"""
		Apply this class's transformation to a single 2D input.
		"""
		x = x.view(1,1, *x.shape)
		x = self.conv(x).view(self.channels, self.h, self.w)
		ret = []
		for channel in x[:]:
			x_series = [None for _ in range (self.rows+1)]
			x_series[0] = channel @ channel 
			
			for i in range(self.rows):
				x_series[i+1] = channel @ x_series[i]
				for j in range(self.cols): ret.append(torch.trace(x_series[i])**(j+1) / torch.numel(channel)**(i+j+1))
		terms = (self.coef * torch.stack(ret).to(x.device).view(self.channels, self.rows, self.cols))
		return terms.sum()

	def stuff_weights(self, weights): assert(0)
	def weight_count(self): assert(0)

	def forward(self, x):
		if len(x.shape) == 2: x = x.view(-1, *x.shape)
		elif len(x.shape) > 3: assert 0
		return torch.stack([self.apply_coef(i) for i in x[:]]).to(x.device)

class ACoef(nn.Module):
	def __init__(self, rows, cols):
		"""
		:param rows: The number of rows in the output matrix
		:param cols: The number of columns in the output matrix.
		"""
		super().__init__()
		self.rows, self.cols = rows, cols
		self.coef = nn.parameter.Parameter(torch.zeros((rows, cols)))
		for x in self.parameters():
			if x.dim() > 1: nn.init.kaiming_normal_(x)

	def stuff_weights(self, weights):
		self.coef.weight = weights
	def weight_count(self):
		return functools.reduce(lambda x, y: x*y, self.coef.shape, 1)

	def forward(self, x):
		# Force all data to be batched if it isn't already.
		if len(x.shape) == 2: x = x.view(-1, *x.shape)
		# But we don't know how to deal with batches-{of-batches}+.
		elif len(x.shape) > 3: assert 0
		
		# A series is the powers of our adjacency matrix(es) X.
		# Compute as many powers as we have rows for each adjacency matrix.
		a_series = [torch.matrix_power(x,i+1) for i in range (1, self.rows+2)]
		# Must swap dims (0,1) since the above code places the batch as dim 1 rather than 0.
		a_series = torch.swapaxes(torch.stack(a_series),0,1).to(x.device)

		# Element wise raise the A series to the correct power, will normalize later.
		# Generator expression performs faster than for loop after profiling.
		powers = list(((a_series[:,i])**(j+1)) for i in range(self.rows) for j in range(self.cols))
		powers = torch.swapaxes(torch.stack(powers), 0,1).to(x.device)

		# Cannot use torch.trace, since that only works on 2d tensors, must roll our own using diag+sum.
		# See: https://discuss.pytorch.org/t/is-there-a-way-to-compute-matrix-trace-in-batch-broadcast-fashion/43866
		traces = torch.diagonal(powers, dim1=-2, dim2=-1).sum(-1)
		traces = traces.view(-1, self.rows, self.cols)
		# The [i,j]'th position is equal to i+j+2. This is the power to which 
		norm_pow_mat = torch.stack(list(torch.arange(0, self.cols)+i+2 for i in range(self.rows))).to(traces.device)
		# Compute the number of elements in an individual graph
		numel = powers.shape[-1]*powers.shape[-2]
		# The normalization for the [i,j]'th entry of each matrix is the number of elements raised to the i+j+2'th power.
		norm =  torch.full(traces.shape, numel).to(traces.device)**norm_pow_mat
		return (self.coef * traces/norm).sum(dim=[-1,-2])

class FACoef(nn.Module):
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

	def stuff_weights(self, weights):
		self.coef.weight = weights
	def weight_count(self):
		return functools.reduce(lambda x, y: x*y, self.coef.shape, 1)

	def forward(self, x):
		
		# Force all data to be batched if it isn't already.
		if len(x.shape) == 2: x = x.view(-1, *x.shape)
		# But we don't know how to deal with batches-{of-batches}+.
		elif len(x.shape) > 3: assert 0
		
		# A series is the powers of our adjacency matrix(es) X.
		# Compute as many powers as we have rows for each adjacency matrix.
		a_series = [torch.matrix_power(x,i+1) for i in range (1, self.rows+2)]
		# Must swap dims (0,1) since the above code places the batch as dim 1 rather than 0.
		a_series = torch.swapaxes(torch.stack(a_series),0,1).to(x.device)

		# Generate the full NxN matrix of 1's.
		_1 = torch.full(x.shape[-2:], 1.0, dtype=x.dtype).to(x.device)
		# Element wise raise the A series to the correct power, will normalize later.
		# Generator expression performs faster than for loop after profiling.
		powers = list((_1@(a_series[:,i])**(j+1)) for i in range(self.rows) for j in range(self.cols))
		powers = torch.swapaxes(torch.stack(powers), 0,1).to(x.device)

		# Cannot use torch.trace, since that only works on 2d tensors, must roll our own using diag+sum.
		# See: https://discuss.pytorch.org/t/is-there-a-way-to-compute-matrix-trace-in-batch-broadcast-fashion/43866
		traces = torch.diagonal(powers, dim1=-2, dim2=-1).sum(-1)
		traces = traces.view(-1, self.rows, self.cols)
		# The [i,j]'th position is equal to i+j+2. This is the power to which 
		norm_pow_mat = torch.stack(list(torch.arange(0, self.cols)+i+2 for i in range(self.rows))).to(traces.device)
		# Compute the number of elements in an individual graph
		numel = powers.shape[-1]*powers.shape[-2]
		# The normalization for the [i,j]'th entry of each matrix is the number of elements raised to the i+j+2'th power.
		norm =  torch.full(traces.shape, numel).to(traces.device)**norm_pow_mat
		return (self.coef * traces/norm).sum(dim=[-1,-2])

class MACoef(nn.Module):
	def __init__(self, graph_size, rows=1, cols=2):
		"""
		:param graph_size: The number of nodes in an input graph.
		:param rows: The number of rows in the output matrix
		:param cols: The number of columns in the output matrix.
		"""
		super().__init__()
		self.rows, self.cols = rows, cols
		self._M = nn.ParameterList([nn.parameter.Parameter(torch.zeros(graph_size, graph_size, dtype=torch.float)) 
			for i in range(rows*cols)])
		self.coef = nn.parameter.Parameter(torch.zeros((rows, cols)))

		for x in self.parameters():
			if x.dim() > 1: nn.init.kaiming_normal_(x)

	def M(self, r, c):
		return self._M[r * self.cols + c]

	def apply_coef(self, x):
		"""
		Apply this class's transformation to a single 2D input.
		"""
		outs = []
		for r in range(self.rows):
			for c in range(self.cols): 
				v = self.M(r,c) @ (x**(c+1))
				outs.append(v.trace()**(r+1))
		return (self.coef * torch.stack(outs).to(x.device).view(self.rows, self.cols)).sum()

	def stuff_weights(self, weights): assert(0)
	def weight_count(self): assert(0)

	def forward(self, x):
		x = x.view(-1, self.graph_size, self.graph_size)
		return torch.stack([self.apply_coef(i) for i in x[:]]).to(x.device)

class SumTerm(nn.Module):
	def __init__(self, mod_list):
		super().__init__()
		self.mod_list = torch.nn.ModuleList(mod_list)

	def apply_coef(self, x):
		"""
		Apply all nested transformations in parallel to a single input.
		"""
		mod_sum = torch.sum(self.mod_list[0].apply_coef(x))
		for _mod in self.mod_list[1:]: mod_sum += torch.sum(_mod.apply_coef(x))
		return mod_sum

	def stuff_weights(self, weights):
		assert len(weights) == self.weight_count()
		for model in self.mod_list:
			model.stuff_weights(weights[:model.weight_count()])
			weights = weights[:model.weight_count()]

	def weight_count(self):
		count = 0
		for model in self.mod_list: count += model.weight_count()
		return count

	def forward(self, x):
		if len(x.shape) == 2: x = x.view(-1, *x.shape)
		elif len(x.shape) > 3: assert 0
		mod_sums = self.mod_list[0](x)
		for _mod in self.mod_list[1:]: mod_sums += _mod(x)
		return torch.sigmoid(mod_sums)

def nearest(prediction, bins):
	best_bins = np.abs(bins - prediction.item())
	return bins[best_bins.argmin()]

def evaluate(net, testloader, dev,  count=None, critertion=nn.MSELoss()):
	total = 0
	loss = 0
	with torch.no_grad():
		for data in testloader:
			images, purity = data
			images, purity = images.to(dev), purity.to(dev)
			outputs = net(images)    
			total += purity.size(0)
			loss += critertion(outputs, purity)

			if count is None: pass
			elif count > 0: count -= len(images)
			elif count <= 0: break
			
	return loss/total

def get_best_config(pure_dir, impure_dir, graph_size, clique_size, net_fn, epochs=100, batch_size=10, dev='cpu', n_splits=2):
	# K-Fold cross validation from: https://www.machinecurve.com/index.php/2021/02/03/how-to-use-k-fold-cross-validation-with-pytorch/
	dataset = graphity.data.FileGraphDataset(pure_dir, impure_dir)
	folds = KFold(n_splits=n_splits, shuffle=True)
	all_loss = []
	print("Generated")
	best_config, best_loss = 0, float("inf")
	for fold, (train_ids, test_ids) in enumerate(folds.split(dataset)):
		criterion = nn.MSELoss()
		net = net_fn()
		optimizer = optim.Adam(net.parameters(), lr=0.001)

		# Augment our pure graphs, so that there are roughly as many pure graphs as there are non-pure graphs
		train_dataset, test_dataset = dataset.split(train_ids)
		
		# Define data loaders for training and testing data in this fold
		trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
		testloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)
		for epoch in range(epochs):  # loop over the dataset multiple times
			running_loss = 0.0
			for i, data in enumerate(trainloader, 0):
				inputs, labels = data
				inputs, labels = inputs.to(dev), labels.float().to(dev)
				optimizer.zero_grad()
				outputs = net(inputs)
				loss = criterion(outputs, labels)
				loss.backward()
				optimizer.step()
				
				running_loss += loss.item()
				if i % 3000 == 2999:    # print every 1000 mini-batches
					print('[%d, %5d] loss: %.3f' %
						(epoch + 1, i + 1, running_loss / 3000))
					running_loss = 0.0
					
		loss = evaluate(net, testloader, dev=dev, critertion=criterion)
		print(f'(k_{clique_size},g_{graph_size})Loss of the network on the {test_dataset.count} graphs: {loss}')
		if loss < best_loss: best_config = net

	return best_config, best_loss, all_loss