import random

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import KFold

import graphity.environment.graph
from graphity.environment.graph.generate import random_adj_matrix, random_pure_graph
import graphity.read_data
import graphity.utils

class Net(nn.Module):
	def __init__(self):
		super().__init__()
		#self.conv1 = nn.Conv2d(3, 6, 5)
		#self.pool = nn.MaxPool2d(2, 2)
		#self.conv2 = nn.Conv2d(6, 16, 5)
		self.fc1 = nn.Linear(20**2, 2, bias=False)
		#self.fc2 = nn.Linear(14, 20)
		#self.fc3 = nn.Linear(1, 2)

	def forward(self, x):
		#x = self.pool(F.relu(self.conv1(x)))
		#x = self.pool(F.relu(self.conv2(x)))
		x = torch.flatten(x, 1) # flatten all dimensions except batch
		#x = F.relu(self.fc1(x))
		#x = F.relu(self.fc2(x))
		x = self.fc1(x)
		return x

class PureGraphDataset(Dataset):
	def __init__(self, count, transform=None, target_transform=None):
		self.count = count
		things = [(0, random_pure_graph(10, 20)) for i in range(count//2)]
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
		image = image.float()
		if self.transform:
			image = self.transform(image)
		if self.target_transform:
			label = self.target_transform(label)
		return image, label


import torch.optim as optim

if __name__ == "__main__":
	transform = transforms.Compose(
	[transforms.ToTensor(),
	transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

	batch_size = 4

	# K-Fold cross validation from: https://www.machinecurve.com/index.php/2021/02/03/how-to-use-k-fold-cross-validation-with-pytorch/
	dataset = PureGraphDataset(count=4000, transform=None)
	folds = KFold(n_splits=10, shuffle=True)

	classes = ('pure', 'not pure')
	print("Generated")
	for fold, (train_ids, test_ids) in enumerate(folds.split(dataset)):
		net = Net()
		criterion = nn.CrossEntropyLoss()
		optimizer = optim.Adam(net.parameters(), lr=0.001)
		# Sample elements randomly from a given list of ids, no replacement.
		train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
		test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
		# Define data loaders for training and testing data in this fold
		trainloader = torch.utils.data.DataLoader(
			dataset, 
			batch_size=10, sampler=train_subsampler)
		testloader = torch.utils.data.DataLoader(
			dataset,
			batch_size=10, sampler=test_subsampler)

		for epoch in range(1):  # loop over the dataset multiple times

			running_loss = 0.0
			for i, data in enumerate(trainloader, 0):
				inputs, labels = data
				optimizer.zero_grad()
				outputs = net(inputs)
				loss = criterion(outputs, labels)
				loss.backward()
				optimizer.step()
				
				running_loss += loss.item()
				if i % 2000 == 1999:    # print every 2000 mini-batches
					print('[%d, %5d] loss: %.3f' %
						(epoch + 1, i + 1, running_loss / 2000))
					running_loss = 0.0
		correct, total = 0, 0
		correct_pred = {classname: 0 for classname in classes}
		total_pred = {classname: 0 for classname in classes}
		with torch.no_grad():
			for data in testloader:
				images, labels = data    
				outputs = net(images)    
				_, predictions = torch.max(outputs, 1)
				total += labels.size(0)
				correct += (predictions == labels).sum().item()
				# collect the correct predictions for each class
				for label, prediction in zip(labels, predictions):
					if label == prediction:
						correct_pred[classes[label]] += 1
					total_pred[classes[label]] += 1
		print(f'Accuracy of the network on the {dataset.count} test images: {100 * correct / total}')
		for classname, correct_count in correct_pred.items():
			accuracy = 100 * float(correct_count) / total_pred[classname]
			print("Accuracy for class {:5s} is: {:.1f} %".format(classname, 
				accuracy))

