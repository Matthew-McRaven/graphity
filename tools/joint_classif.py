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

def trace_series(adj, rows, cols):
	tr_ret = torch.zeros((rows, cols))
	full_ret = torch.zeros((rows, cols))
	_1 = torch.full(adj.shape, 1.0, dtype=torch.float)
	
	a_series = [None for _ in range (rows+1)]
	a_series[0] = adj.matmul(adj)
	for i in range(rows):	
		a_series[i+1] = adj.matmul(a_series[i])
		for j in range(cols):
			tr_ret[i,j] = torch.trace(a_series[i])**(j+1) / torch.numel(adj)**(i+j+1)
			full_ret[i,j] = torch.trace(a_series[i].float().matmul(_1))**(j+1) / torch.numel(adj)**(i+j+2)
	return torch.stack((tr_ret, full_ret))

class MHack(nn.Module):
	def __init__(self, graph_size, rows=1, cols=2, terms=None):
		super().__init__()
		self.rows, self.cols = rows, cols
		self._M = nn.ParameterList([nn.parameter.Parameter(torch.zeros(graph_size, graph_size, dtype=torch.float)) 
			for i in range(rows*cols)])
		self.coef = nn.parameter.Parameter(torch.zeros((cols, rows)))
		self.terms = terms

		for x in self.parameters():
			if x.dim() > 1: nn.init.kaiming_normal_(x)

	def M(self, r, c):
		return self._M[r * self.cols + c]

	def forward(self, x):
		outs = []
		for i in x[:]:
			for r in range(self.rows):
				for c in range(self.cols): outs.append(self.M(r,c).matmul(i**(c+1)).trace()**r)
			
		ret = torch.stack(outs).view(x.shape[0], self.rows, self.cols)

		
		ret = self.coef*ret
		ret = ret.sum(dim=(1,2))

		ret = torch.sigmoid(ret)
		return ret

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
	best_weights, best_accuracy = 0, 0
	for fold, (train_ids, test_ids) in enumerate(folds.split(dataset)):
		net = MHack(graph_size, rows=reduction[0], cols=reduction[1])
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
				labels = labels.float()
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
				outputs = net(images)    
				predictions = outputs.round()
				total += labels.size(0)
				correct += (predictions == labels).sum().item()
				# collect the correct predictions for each class
				for label, prediction in zip(labels, predictions):
					if label == prediction: correct_pred[classes[label]] += 1
					total_pred[classes[label]] += 1
		print(f'Accuracy of the network on the {dataset.count} test images: {100 * correct / total}')
		if correct/total > best_accuracy: best_weights = (net._M, net.coef)
		for classname, correct_count in correct_pred.items():
			accuracy = 100 * float(correct_count) / total_pred[classname]
			print("Accuracy for class {:5s} is: {:.1f} %".format(classname, accuracy))
	return best_weights, correct/total

def weight_mul(item, weight, reduction):
	outs = []
	rows, cols = reduction[0], reduction[1]
	def id(r,c):
		return r*cols + c
	for r in range(reduction[0]):
		for c in range(reduction[1]):
			outs.append(weight[0][id(r,c)].matmul(item.double()).trace())
	ret = torch.stack(outs).view(rows, cols)
	ret = (weight[1] * ret).sum()
	return ret.detach_()

def classify(weight, clique_size, graph_size, count, reduction=None):
	things = [random_pure_graph(clique_size, graph_size) for i in range(count)]
	p_edge = things[0].float().mean()
	not_things = [random_adj_matrix(things[0].shape[0], p=p_edge) for i in range(1*len(things))]
	print("Regenerated")
	data = [weight_mul(i, weight, reduction).view(-1).numpy() for i in things+not_things]
	target = len(things)*[1]+ len(not_things)*[2]
	clf = linear_model.SGDClassifier()
	X_train, X_test, y_train, y_test = train_test_split(
		data, target, test_size=0.5, shuffle=True)

	clf.fit(X_train, y_train)
	

	predicted = clf.predict(X_test)
	print(f"Classification report for classifier {clf}:\n", f"{metrics.classification_report(y_test, predicted)}\n")
	print(f"Confusion matrix:\n{metrics.plot_confusion_matrix(clf, X_test, y_test).confusion_matrix}")



if __name__ == "__main__":
	# Hyper parameters of the system. Clique size, max graph size, and the (rows, columns) in the trace matrix.
	c, g = 3, 10
	reduction = (2,1)

	x_s = range(4,g)
	y_s = [None for _ in x_s]
	
	###################################################
	# Must increase count for stat. signifcant result #
	###################################################
	for i in range(4,g):
		weight, y_s[i-4] = get_weight(c, i, reduction=reduction, count=10000)
		weight = [i.double() for i in weight] 
		classify(weight, c, i, 1000, reduction=reduction)
	fig = plt.figure()
	ax = plt.axes()
	ax.plot(x_s, y_s)
	plt.show()


