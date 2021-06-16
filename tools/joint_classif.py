from functools import reduce
from itertools import count
import random

import matplotlib.pyplot as plt
from networkx.algorithms import clique
import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import KFold
from sklearn import datasets, svm, metrics, decomposition, neighbors, linear_model
from sklearn.model_selection import train_test_split

import graphity.environment.graph
from graphity.environment.graph.generate import random_adj_matrix, random_pure_graph
import graphity.read_data
import graphity.utils

dev = "cpu"
if torch.cuda.is_available(): dev = "cuda"

def conv_output_shape(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
    from math import floor
    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size)
    h = floor( ((h_w[0] + (2 * pad) - ( dilation * (kernel_size[0] - 1) ) - 1 )/ stride) + 1)
    w = floor( ((h_w[1] + (2 * pad) - ( dilation * (kernel_size[1] - 1) ) - 1 )/ stride) + 1)
    return h, w
class Raw(nn.Module):
	def __init__(self, graph_size):
		super().__init__()
		self._M = nn.parameter.Parameter(torch.zeros((1, graph_size, graph_size)))
		for x in self.parameters():
			if x.dim() > 1: nn.init.kaiming_normal_(x)

	def apply_coef(self, x):
		return self._M @ x

	def forward(self, x):
		return torch.stack([self.apply_coef(i) for i in x[:]]).to(dev)

class Conv(nn.Module):
	def __init__(self, graph_size, rows, cols, channels=1, k=6):
		super().__init__()
		self.rows, self.cols = rows, cols
		self.coef = nn.parameter.Parameter(torch.zeros((rows, cols)))
		self.conv = torch.nn.Conv2d(1, channels, (k,k))
		hw = conv_output_shape((graph_size, graph_size), (k,k))
		self.lin = torch.nn.Linear(hw[0]*hw[1]*channels, rows*cols)
		for x in self.parameters():
			if x.dim() > 1: nn.init.kaiming_normal_(x)

	def apply_coef(self, x):
		x = x.view(1,1, *x.shape)
		x = self.conv(x)
		x = self.lin(x.view(-1)).view(self.rows, self.cols)
		return self.coef * x


	def forward(self, x):
		return torch.stack([self.apply_coef(i) for i in x[:]]).to(dev)

class ConvTrace(nn.Module):
	def __init__(self, graph_size, rows, cols, channels=5, k=6):
		super().__init__()
		self.rows, self.cols, self.channels = rows, cols, channels
		self.coef = nn.parameter.Parameter(torch.zeros((channels, rows, cols)))
		self.conv = torch.nn.Conv2d(1, channels, (k,k))
		hw = conv_output_shape((graph_size, graph_size), (k,k))
		self.h, self.w = hw[0], hw[1]
		for x in self.parameters():
			if x.dim() > 1: nn.init.kaiming_normal_(x)

	def apply_coef(self, x):
		
		x = x.view(1,1, *x.shape)
		x = self.conv(x).view(self.channels, self.h, self.w)
		ret = []
		for channel in x[:]:
			x_series = [None for _ in range (self.rows+1)]
			x_series[0] = channel @ channel 
			
			for i in range(self.rows):
				x_series[i+1] = channel @ x_series[i]
				for j in range(self.cols): ret.append(torch.trace(x_series[i])**(j+1) / torch.numel(channel)**(i+j+1))
		terms = (self.coef * torch.stack(ret).view(self.channels, self.rows, self.cols))
		return terms.sum()


	def forward(self, x):
		return torch.stack([self.apply_coef(i) for i in x[:]]).to(dev)
class ACoef(nn.Module):
	def __init__(self, rows, cols):
		super().__init__()
		self.rows, self.cols = rows, cols
		self.coef = nn.parameter.Parameter(torch.zeros((rows, cols)))
		for x in self.parameters():
			if x.dim() > 1: nn.init.kaiming_normal_(x)

	def apply_coef(self, x):
		ret = []
		a_series = [None for _ in range (self.rows+1)]
		a_series[0] = x @ (x)
		for i in range(self.rows):	
			a_series[i+1] = x.matmul(a_series[i])
			for j in range(self.cols): ret.append(torch.trace(a_series[i])**(j+1) / torch.numel(x)**(i+j+1))
		terms = (self.coef * torch.stack(ret).view(self.rows, self.cols))
		return terms.sum()

	def forward(self, x):
		return torch.stack([self.apply_coef(i) for i in x[:]]).to(dev)

class FACoef(nn.Module):
	def __init__(self, rows, cols):
		super().__init__()
		self.rows, self.cols = rows, cols
		self.coef = nn.parameter.Parameter(torch.zeros((rows, cols)))
		for x in self.parameters():
			if x.dim() > 1: nn.init.kaiming_normal_(x)

	def apply_coef(self, x):
		ret = []
		_1 = torch.full(x.shape, 1.0, dtype=torch.float)
		a_series = [None for _ in range (self.rows+1)]
		a_series[0] = x.matmul(x)
		for i in range(self.rows):	
			a_series[i+1] = x.matmul(a_series[i])
			for j in range(self.cols): ret.append(torch.trace(_1 @ (a_series[i].float()))**(j+1) / torch.numel(x)**(i+j+2))
		return (self.coef * torch.stack(ret).view(self.rows, self.cols)).sum()
		
	def forward(self, x):
		ret = torch.stack([self.apply_coef(i) for i in x[:]])
		return ret 

class MACoef(nn.Module):
	def __init__(self, graph_size, rows=1, cols=2):
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
		outs = []
		for r in range(self.rows):
			for c in range(self.cols): 
				v = self.M(r,c) @ (x**(c+1))
				outs.append(v.trace()**(r+1))
		return (self.coef * torch.stack(outs).to(dev).view(self.rows, self.cols)).sum()

	def forward(self, x):
		return torch.stack([self.apply_coef(i) for i in x[:]]).to(dev)

class SumTerm(nn.Module):
	def __init__(self, mod_list):
		super().__init__()
		self.mod_list = torch.nn.ModuleList(mod_list)

	def apply_coef(self, x):
		mod_sum = torch.sum(self.mod_list[0].apply_coef(x))
		for _mod in self.mod_list[1:]: mod_sum += torch.sum(_mod.apply_coef(x))
		return mod_sum

	def forward(self, x):
		return torch.sigmoid(torch.stack([self.apply_coef(i) for i in x[:]]))
class PureGraphDataset(Dataset):
	def __init__(self, clique_size, graph_size, count, transform=None, target_transform=None):
		self.count = count
		things = [(0, random_pure_graph(clique_size, graph_size)) for i in range(count//2)]
		p_edge = things[0][1].float().mean()
		not_things = [(1, random_adj_matrix(things[0][1].shape[0], p=p_edge)) for i in range(count-len(things))]
		#things = things + [i for i in not_things if graphity.utils.is_pure(i[1])]
		#not_things = [i for i in not_things if not graphity.utils.is_pure(i[1])]
		self.data = things+not_things
		random.shuffle(self.data)
		self.transform = transform
		self.target_transform = target_transform

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		label, image = self.data[idx]
		if self.transform:
			image = self.transform(image)
		if self.target_transform:
			label = self.target_transform(label)
		return image.float(), label

def get_weight(clique_size, graph_size, epochs=4, batch_size=10, reduction=None, count=20000):
	# K-Fold cross validation from: https://www.machinecurve.com/index.php/2021/02/03/how-to-use-k-fold-cross-validation-with-pytorch/
	dataset = PureGraphDataset(clique_size, graph_size, count=count, transform=None)
	folds = KFold(n_splits=2, shuffle=True)

	classes = ('pure', 'not pure')
	print("Generated")
	best_config, best_accuracy = 0, 0
	for fold, (train_ids, test_ids) in enumerate(folds.split(dataset)):
		terms = []
		#terms.append(Raw(graph_size))
		#terms.append(MACoef(graph_size, rows=reduction[0], cols=reduction[1]))
		#terms.append(ACoef(rows=reduction[0], cols=reduction[1]))
		#terms.append(FACoef(rows=reduction[0], cols=reduction[1]))
		terms.append(ConvTrace(graph_size, rows=reduction[0], cols=reduction[1]))
		#terms.append(Conv(graph_size, rows=reduction[0], cols=reduction[1]))
		net = SumTerm(terms)
		net = net.to(dev)
		criterion = nn.BCELoss()
		optimizer = optim.Adam(net.parameters(), lr=0.001)
		# Sample elements randomly from a given list of ids, no replacement.
		train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
		test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
		# Define data loaders for training and testing data in this fold
		trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_subsampler)
		testloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=test_subsampler)

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
				if i % 1000 == 999:    # print every 1000 mini-batches
					print('[%d, %5d] loss: %.3f' %
						(epoch + 1, i + 1, running_loss / 2000))
					running_loss = 0.0

		correct, total = 0, 0
		correct_pred, total_pred = {classname: 0 for classname in classes}, {classname: 0 for classname in classes}
		with torch.no_grad():
			for data in testloader:
				images, labels = data
				inputs, labels = inputs.to(dev), labels.to(dev)   
				outputs = net(images)    
				predictions = outputs.round()
				total += labels.size(0)
				correct += (predictions == labels).sum().item()
				# collect the correct predictions for each class
				for label, prediction in zip(labels, predictions):
					if label == prediction: correct_pred[classes[label]] += 1
					total_pred[classes[label]] += 1
		print(f'Accuracy of the network on the {dataset.count} test images: {100 * correct / total}')
		if correct/total > best_accuracy: best_config = net
		for classname, correct_count in correct_pred.items():
			accuracy = 100 * float(correct_count) / total_pred[classname]
			print("Accuracy for class {:5s} is: {:.1f} %".format(classname, accuracy))
	return best_config, correct/total

def classify(config, clique_size, graph_size, count, reduction=None):
	#print([i for i in config.parameters()])
	things = [random_pure_graph(clique_size, graph_size) for i in range(count)]
	p_edge = things[0].float().mean()
	not_things = [random_adj_matrix(things[0].shape[0], p=p_edge) for i in range(1*len(things))]
	print("Regenerated")
	data = [config.apply_coef(i.float()).detach().view(-1).numpy() for i in things+not_things]
	target = len(things)*[1]+ len(not_things)*[2]
	clf = svm.SVC()
	X_train, X_test, y_train, y_test = train_test_split(
		data, target, test_size=0.5, shuffle=True)

	clf.fit(X_train, y_train)
	

	predicted = clf.predict(X_test)
	print(f"Classification report for classifier {clf}:\n", f"{metrics.classification_report(y_test, predicted)}\n")
	print(f"Confusion matrix:\n{metrics.confusion_matrix(y_test, clf.predict(X_test))}")



if __name__ == "__main__":
	# Hyper parameters of the system. Clique size, max graph size, and the (rows, columns) in the trace matrix.
	c, g = 3, 20
	min_g = 6
	reduction = (4,3)

	x_s = range(min_g, g)
	y_s = [None for _ in x_s]
	
	###################################################
	# Must increase count for stat. signifcant result #
	###################################################
	for i in range(min_g,g):
		config, y_s[i-min_g-1] = get_weight(c, i, reduction=reduction, count=1000)
		classify(config, c, i, 1000, reduction=reduction)
	fig = plt.figure()
	ax = plt.axes()
	ax.plot(x_s, y_s[:])
	plt.show()


