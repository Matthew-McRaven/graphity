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
		return self.coef * x

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

	def apply_coef(self, x):
		"""
		Apply this class's transformation to a single 2D input.
		"""
		ret = []
		a_series = [None for _ in range (self.rows+1)]
		a_series[0] = x @ (x)
		for i in range(self.rows):	
			a_series[i+1] = x.matmul(a_series[i])
			for j in range(self.cols): ret.append(torch.trace(a_series[i])**(j+1) / torch.numel(x)**(i+j+1))
		terms = (self.coef * torch.stack(ret).to(x.device).view(self.rows, self.cols))
		return terms.sum()

	def stuff_weights(self, weights):
		self.coef.weight = weights
	def weight_count(self):
		return functools.reduce(lambda x, y: x*y, self.coef.shape, 1)

	def forward(self, x):
		if len(x.shape) == 2: x = x.view(-1, *x.shape)
		elif len(x.shape) > 3: assert 0
		return torch.stack([self.apply_coef(i) for i in x[:]]).to(x.device)

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

	def apply_coef(self, x):
		"""
		Apply this class's transformation to a single 2D input.
		"""
		ret = []
		_1 = torch.full(x.shape, 1.0, dtype=torch.float).to(x.device)
		a_series = [None for _ in range (self.rows+1)]
		a_series[0] = x.matmul(x)
		for i in range(self.rows):	
			a_series[i+1] = x.matmul(a_series[i])
			for j in range(self.cols): ret.append(torch.trace(_1 @ (a_series[i].float()))**(j+1) / torch.numel(x)**(i+j+2))
		return (self.coef * torch.stack(ret).to(x.device).view(self.rows, self.cols)).sum()

	def stuff_weights(self, weights):
		self.coef.weight = weights
	def weight_count(self):
		return functools.reduce(lambda x, y: x*y, self.coef.shape, 1)

	def forward(self, x):
		if len(x.shape) == 2: x = x.view(-1, *x.shape)
		elif len(x.shape) > 3: assert 0
		return torch.stack([self.apply_coef(i) for i in x[:]]).to(x.device)

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
		return torch.sigmoid(torch.stack([self.apply_coef(i) for i in x[:]]))

def evaluate(net, testloader, dev):
	classes = ('pure', 'not pure')
	correct, total = 0, 0
	correct_pred, total_pred = {classname: 0 for classname in classes}, {classname: 0 for classname in classes}
	with torch.no_grad():
		for data in testloader:
			images, labels = data
			images, labels = images.to(dev), labels.to(dev)
			outputs = net(images)    
			predictions = outputs.round()
			total += labels.size(0)
			correct += (predictions == labels).sum().item()
			# collect the correct predictions for each class
			for label, prediction in zip(labels, predictions):
				if label == prediction: correct_pred[classes[label]] += 1
				total_pred[classes[label]] += 1
	return correct/total, (correct_pred, total_pred)

def get_best_config(pure_dir, impure_dir, graph_size, clique_size, net_fn, epochs=100, batch_size=10, dev='cpu', n_splits=2):
	# K-Fold cross validation from: https://www.machinecurve.com/index.php/2021/02/03/how-to-use-k-fold-cross-validation-with-pytorch/
	dataset = graphity.data.FileGraphDataset(pure_dir, impure_dir)
	folds = KFold(n_splits=n_splits, shuffle=True)
	all_accuracy = []
	print("Generated")
	best_config, best_accuracy = 0, 0
	for fold, (train_ids, test_ids) in enumerate(folds.split(dataset)):
		criterion = nn.BCELoss()
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
				if i % 10000 == 9999:    # print every 1000 mini-batches
					print('[%d, %5d] loss: %.3f' %
						(epoch + 1, i + 1, running_loss / 2000))
					running_loss = 0.0
					
		accuracy, (correct_pred, total_pred) = evaluate(net, testloader, dev=dev)
		all_accuracy.append(accuracy)
		print(f'(k_{clique_size},g_{graph_size})Accuracy of the network on the {test_dataset.count} test images: {100 * accuracy}')
		for classname, correct_count in correct_pred.items():
			accuracy = 100 * float(correct_count) / total_pred[classname]
			print("Accuracy for class {:5s} is: {:.1f} %".format(classname, accuracy))
		if accuracy > best_accuracy: best_config = net

	return best_config, best_accuracy, all_accuracy