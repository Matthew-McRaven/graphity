import more_itertools
from numpy.core.fromnumeric import argmin
import torch
import torch.optim
from torch import nn

import graphity.task.sampler
import graphity.environment.reward
import graphity.grad.grad

#Predict the value / energy of a given graph
class ValuePredictor(nn.Module):
    def __init__(self, graph_size, layer_list):
        super(ValuePredictor, self).__init__()
        self.graph_size = graph_size
        self.linear_unit = None
        dropout = 0.1
        self.input_dimensions = (graph_size, graph_size)
        self._input_size = graph_size**2

        # Build linear layers from input defnition.
        linear_layers = []
        previous = self._input_size
        for index,layer in enumerate(layer_list):
            linear_layers.append(nn.Linear(previous, layer))
            linear_layers.append(nn.LeakyReLU())
            # We have an extra component at the end, so we can dropout after every layer.
            linear_layers.append(nn.Dropout(dropout))
            previous = layer
        linear_layers.append(nn.Linear(previous, 1))
        self.output_dimension = (1, )
        self.linear_layers = nn.Sequential(*linear_layers)

        self.optim = torch.optim.Adam(self.parameters(), lr=0.001, weight_decay=0.2)

        # Initialize NN
        for x in self.parameters():
            if x.dim() > 1:
                nn.init.kaiming_normal_(x)

    def forward(self, graph):
        graph = graph.float()
        return self.linear_layers(graph.view(-1, self.graph_size**2))

# Predict the gradient of a given graph.
class GradPredictor(nn.Module):
    def __init__(self, graph_size, layer_list):
        super(GradPredictor, self).__init__()
        self.graph_size = graph_size
        self.linear_unit = None
        dropout = 0.1
        self.input_dimensions = (graph_size, graph_size)
        self._input_size = graph_size**2

        # Build linear layers from input defnition.
        linear_layers = []
        previous = self._input_size
        for index,layer in enumerate(layer_list):
            linear_layers.append(nn.Linear(previous, layer))
            linear_layers.append(nn.LeakyReLU())
            # We have an extra component at the end, so we can dropout after every layer.
            linear_layers.append(nn.Dropout(dropout))
            previous = layer
        linear_layers.append(nn.Linear(previous, graph_size**2))
        self.output_dimension = (graph_size, graph_size)
        self.linear_layers = nn.Sequential(*linear_layers)

        self.optim = torch.optim.Adam(self.parameters(), lr=0.001, weight_decay=0.2)

        # Initialize NN
        for x in self.parameters():
            if x.dim() > 1:
                nn.init.kaiming_normal_(x)

    def forward(self, graph):
        graph = graph.float()
        return self.linear_layers(graph.view(-1, self.graph_size**2)).view(-1, *self.output_dimension)
# Transformations to make numbers "nice" for neural nets
def forward_remap(x):
    return torch.exp(x)
def backward_remap(x):
    return torch.log(x)

# Generate (input, output) pairs for a valuation task.
def value_generate_batch(H, graph_size, count=100):
    sampler = graphity.task.graphity.task.sampler.RandomSampler(graph_size)
    samples = [sampler.sample() for _ in range(count)]
    energies = [backward_remap(H(sample)) for sample in samples]
    return torch.stack(samples), torch.stack(energies).float()


def value_eval_batch(nn, samples, energies, criterion):
    output = nn.forward(samples)
    return criterion(output, energies)
# Generate (input, output) pairs for a gradient task
def grad_generate_batch(H, graph_size, count=100):
    sampler = graphity.task.graphity.task.sampler.RandomSampler(graph_size)
    samples = [sampler.sample() for _ in range(count)]
    energies = [graphity.grad.grad.graph_gradient(sample, H) for sample in samples]
    return torch.stack(samples), torch.stack(energies).float()

def grad_eval_batch(nn, samples, energies, criterion):
    output = nn.forward(samples)
    return criterion(output, energies)

# Train a NN to be good at valuation.
def value_train_loop(hypers, nn, criterion=torch.nn.MSELoss(reduction='sum')):
    for epoch in range(hypers['epochs']):
        nn.train()
        train_losses, test_losses = [], []
        for batch in range(hypers['batch_train']):
            graphs, labels = value_generate_batch(hypers['H'], hypers['n'])
            loss = value_eval_batch(nn, graphs, labels, criterion)
            loss.backward()
            train_losses.append(loss.item())
            nn.optim.step(), nn.zero_grad()

        nn.eval()
        for batch in range(hypers['batch_test']):
            graphs, labels = value_generate_batch(hypers['H'], hypers['n'])
            loss = value_eval_batch(nn, graphs, labels, criterion)
            test_losses.append(loss.item())
        print(f"Train loss is {sum(train_losses) / len(train_losses):.4E}.\nTest loss is {sum(test_losses)/len(test_losses):.4E}.\n")
        graph, energy = value_generate_batch(hypers['H'], hypers['n'], 1)
        print(graph, forward_remap(energy), forward_remap(nn(graph)))

# Train a network to be good at taking gradient.
def grad_train_loop(hypers, nn, criterion=torch.nn.MSELoss(reduction='sum')):
    for epoch in range(hypers['epochs']):
        nn.train()
        train_losses, test_losses = [], []
        for batch in range(hypers['batch_train']):
            graphs, labels = grad_generate_batch(hypers['H'], hypers['n'])
            loss = grad_eval_batch(nn, graphs, labels, criterion)
            loss.backward()
            train_losses.append(loss.item())
            nn.optim.step(), nn.zero_grad()

        nn.eval()
        for batch in range(hypers['batch_test']):
            graphs, labels = grad_generate_batch(hypers['H'], hypers['n'])
            loss = grad_eval_batch(nn, graphs, labels, criterion)
            test_losses.append(loss.item())
        print(f"Train loss is {sum(train_losses) / len(train_losses):.4E}.\nTest loss is {sum(test_losses)/len(test_losses):.4E}.\n")
        #graph, grad = grad_generate_batch(hypers['H'], hypers['n'], 1)
        #print(f"{graph}\n{grad.view(args.n, args.n).detach().numpy()}\n {nn(graph).view(args.n, args.n).detach().numpy()}\n")
    