import networkx as nx
import numpy as np
import torch


def torchize(maybe_tensor, device):
	"""
	Massage something that looks like a tensor, ndarray to a tensor on the correct device.
	If given an iterable of those things, recurssively attempt to massage them into tensors.
	Returns a tensor with the same values as maybe_tensor.

	:param maybe_tensor: An array-like object.
	:param device: A valid torch device on whihc tensors are to be stored.
	"""
	# Sometimes an object is a torch tensor that just needs to be moved to the right device
	if torch.is_tensor(maybe_tensor): return maybe_tensor.to(device)
	# Sometimes it is an ndarray
	elif isinstance(maybe_tensor, np.ndarray): return torch.tensor(maybe_tensor).to(device) # type: ignore
	# Maybe a simple iterable of things we need to move to torch, like a cartesian product of ndarrays.
	elif isinstance(maybe_tensor, (list, tuple)): return [torchize(item, device) for item in maybe_tensor]
	elif maybe_tensor is None: return None
	else: raise NotImplementedError(f"Don't understand the datatype of {type(maybe_tensor)}")

def is_matrix(tensor):
	"""
	Check if the tensor is 2D. 
	If so, it is a matrix.

	:param tensor: A torch.Tensor of unknown shape.
	"""
	return len(tensor.shape) == 2


def is_square(tensor):
	"""
	If we have a tensor of matricies, the last two dims correspond to m x n of the matrix.
	For the tensor to be square, m==n.

	:param tensor: A torch.Tensor of unknown shape.
	"""
	return tensor.shape[-1] == tensor.shape[-2]

def is_pure(tensor):
	"""
	Warning!: This requires solving an NP hard problem. 
	This may take exponential time. 
	Please pass in small graphs for your own sake.

	Determines if the input graph is pure.
	For a graph to be pure, the maximal clique number at every site must be the same size.

	:param tensor: A torch.Tensor containing all {0, 1}.
	This tensor's maximal clique size and purity status is not known.
	"""
	G = nx.from_numpy_matrix(tensor.cpu().numpy())
	sizes = nx.node_clique_number(G, [i for i in range(len(G))])
	# Must iterate over values, sine sizes is a dict.
	return all(x == sizes[0] for x in sizes.values())

# 
def is_symmetric(tensor):
	"""
	Return if all matricies in the tensor are symmetric about their diagonal.
	"""
	# Maybe batching has has given us more than 3 dims. If so, flatten it.
	tensor = tensor.view(-1, tensor.shape[-1], tensor.shape[-2])
	# iterate over all matricies, and all indicies
	for k in range(tensor.shape[0]):
		for i in range(tensor.shape[1]):
			# We can start at i+1 rather than 0 because if j were less than i,
			# we would already have check that index in a previous iteration over i.
			# No need to start at i, since diagonal can't affect symmetry.
			for j in range(i+1, tensor.shape[2]):
				if tensor[k,i,j] != tensor [k,j,i]:
					# Abort early, saving us runtime.
					return False
	# Proved for all pairs <i, j> where i<j, G[i,j]=G[k,i], i.e. graph is symmetric.
	return True

# Check if a tensor contains only the values 0 or 1, a requirement for adjacency matricies.
def all_zero_one(tensor):
	# Flatten to 1d for ease
	tensor = tensor.view(-1)
	# Perform elementwise equality to int(0) or int (1). This yields a tensor.
	# Then, perform a reduction that and's together all elements and returns a single element tensor.
	# Extrat the truthy value from tensor with .item()
	x = torch.all(torch.eq(tensor, 1) | torch.eq(tensor, 0)).item()
	return x

def is_adj_matrix(tensor):
	"""
	A tensor can be considered a adjacency matrix if it has 2 dimensions
	of the same size. Additionally, it must be symmetric with all entries in {0, 1}

	:param tensor: A torch.Tensor containing all {0, 1}.
	"""
	return (is_matrix(tensor) and is_square(tensor) 
			and is_symmetric(tensor) and all_zero_one(tensor))

def print_adj_tensor_as_graph(tensor, name="failing.png"):
	# Must view(...) changes the dimensions of the reward tensor from 1xnxn to nxn.
	# Detach deletes any stored gradient information  (important when using machine learning!)
	# CPU forces tensor to CPU, which is required to convert from a tensor to a numpy array.
	adj = tensor.detach().cpu().numpy()
	# Convert from adjacency matrix to NetworkX object.
	# See documentation for information about the library:
	#    https://networkx.github.io/documentation/stable/
	#    https://networkx.github.io/
	#    https://pypi.org/project/networkx/
	# If you have questions about what algorithmsare implemented on these graphs, see:
	#    https://networkx.github.io/documentation/stable/reference/index.html
	as_graph = nx.from_numpy_matrix(adj)

	# Drawing example taken from:
	#    https://networkx.github.io/documentation/latest/auto_examples/drawing/plot_weighted_graph.html
	pos = nx.spring_layout(as_graph) 
	# Draw nodes & edges.
	nx.draw_networkx_nodes(as_graph, pos, node_size=700)
	nx.draw_networkx_edges(as_graph, pos, width=6)
	# Render the graph to the screen.
	plt.axis("off")
	plt.savefig(name)