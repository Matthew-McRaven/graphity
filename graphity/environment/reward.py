import torch

# Implement the hamiltonian discussed with Betre on 20201015
class ASquaredD:
    def __init__(self, d):
        self.d = d
    
    # Allow the class to be called like a function.
    def __call__(self, adj):
        # Force all tensors to be batched.
        if len(adj.shape) == 2:
            adj = adj.view(1,*adj.shape)
        # At this time, I (MM) don't know how matmul will work in 4+ dims.
        # We will fiure this out when it becomes useful.
        elif len(adj.shape) > 3:
            assert False and "Batched input can have at most 3 dimensions" 

        # For each matrix in the batch, compute the adjacency matrix^2.
        temp = torch.matmul(adj, adj)
        temp = torch.sub(temp, self.d)
        # Construct a matrix only containing diagonal
        diag = temp.diagonal(dim1=-2,dim2=-1)
        temp_diag = (diag)
        # Subtract out diagonal, so diagonal is 0.
        temp -= temp_diag
        
        # Sum over the last two dimensions, leaving us with a 1-d array of values.
        return torch.sum(temp.pow(2), (1,2))

