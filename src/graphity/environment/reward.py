import itertools
import functools

import torch
import torch.distributions
import numpy as np

# Implement the hamiltonian discussed with Betre on 20201015
class ASquaredD:
    decomposable = False
    def __init__(self, d, keep_diag=False):
        self.d = d
        self.keep_diag = keep_diag

    @staticmethod
    @functools.lru_cache(1000)
    def compute(adj, d, keep_diag):
        # For each matrix in the batch, compute the adjacency matrix^2.
        temp = torch.matmul(adj, adj) - d
        # Perform element-wise square.
        temp = temp.pow(2)
        # Only mask out diagonal if required.
        if not keep_diag:
            # Construct a matrix only containing diagonal
            n = temp.shape[1]
            diag = (temp.diagonal(dim1=-2,dim2=-1).view(-1) * torch.eye(n, n).view(temp.shape[1:])).long()
            #print(adj, "\n", temp)
            # Subtract out diagonal, so diagonal is 0.
            temp -= diag

        # Sum over the last two dimensions, leaving us with a 1-d array of values.
        # Sum over all non-diagonals.
        return torch.sum(temp, (1,2)) / 2

    # Allow the class to be called like a function.
    def __call__(self, adj):
        # Force all tensors to be batched.
        if len(adj.shape) == 2:
            adj = adj.view(1,*adj.shape)
        # At this time, I (MM) don't know how matmul will work in 4+ dims.
        # We will fiure this out when it becomes useful.
        elif len(adj.shape) > 3:
            assert False and "Batched input can have at most 3 dimensions" 
        return self.compute(adj, self.d, self.keep_diag)

# ASquaredD has explodes in high dimensions.
# Taking the log of this number reduces growth to be more like N**2 rather than a**n**2.
class LogASquaredD(ASquaredD):
    def __init__(self, d, **kwargs):
        super(LogASquaredD, self).__init__(d, **kwargs)
    def __call__(self, adj):
        return np.log(super(LogASquaredD, self).__call__(adj))

# For *really* high dimensions, you may need nested logs.
# However, this will torpedo the ablity to dilineate between states of
# of similiar (but not identical) energy levels.
class NestedLogASquaredD(ASquaredD):
    def __init__(self, d, nesting, **kwargs):
        super(NestedLogASquaredD, self).__init__(d, **kwargs)
        self.nesting = nesting

    def __call__(self, adj):
        start = super(NestedLogASquaredD, self).__call__(adj)
        for i in range(self.nesting):
            start = np.log(start)
        return start


##########################
# Spin Glass Hamiltonian #
##########################
class AbstractSpinGlassHamiltonian():
    decomposable = True
    # "J" encapsulates the interaction between any two particles.
    def __init__(self, J):
        self.J = J


    def compute_glass(self, spins, J):      
        energy = self._contribution(spins, J).sum()

        # Energy is negative (Dr. Betre 20200105), but we need +'ve number to take logs.
        # And we need a -'ve number to minimize (for gradient descent).
        # So, in order to turn this number back into a "real" energy, compute
        #   -e^|energy|
        energy = self.normalize(energy)
        return energy

    def normalize(self, energy):
        ret =  -energy
        assert not torch.isinf(ret).any()

        return ret

    @staticmethod
    @functools.lru_cache(1000)
    def _contribution(spins, J):
        local_spins = spins.clone()
        # Spins ∈ {-1, 1}, but adjacency's are ∈ {0, 1}
        local_spins[local_spins==0] = -1
        contribution = torch.zeros(local_spins.shape, dtype=torch.float32)
        # Unpack size of i,j dimensions, respectively
        dim_i, dim_j = local_spins.shape

        # Iterate over cartesian product of all index pairs.
        # We know a priori what the spin_ij, and spin_lm are, but we need to be told by
        # our subclass how to compute J, so defer to self.J
        for (i,j) in itertools.product(range(dim_i), range(dim_j)):
            # Vectorize computation by computing all (l,m) pairs at once.
            contribution[i,j] = (local_spins[i,j] * local_spins * J[i,j]).double().sum()
        return contribution

    @staticmethod
    def _fast_toggle(spins, contribution, J, toggle):
        # Unpack size of i,j dimensions, respectively
        toggle_i, toggle_j = toggle
        contribution -= 2 * -spins * J[:,:, toggle_i, toggle_j]
        contribution[toggle_i, toggle_j] *= -1
        return contribution

    def fast_toggle(self, spins, contribution, toggle):
        self.J(spins)
        
        return self._fast_toggle(spins, contribution, self.J_Mat, toggle)

    def contribution(self, spins):
        self.J(spins.shape)
        return self._contribution(spins, self.J_Mat)

    def __call__(self, spins):
        self.J(spins.shape)
        return self.compute_glass(spins, self.J_Mat)


# Nearest-neighbor interactions only.
class IsingHamiltonian(AbstractSpinGlassHamiltonian):
    # Simple lookup on pre-computed
    def _J(self, size):
        if self.J_Mat is None:
            dim_i, dim_j = size
            self.J_Mat = torch.zeros((dim_i, dim_j, dim_i, dim_j), dtype=torch.float32)
            # Fill in the "adjacency" matrix for the ising model.
            for (i,j) in itertools.product(range(dim_i), range(dim_j)):
                self.J_Mat[i,j, i-1,j] = 1.
                self.J_Mat[i,j, i,j-1] = 1.
                self.J_Mat[i,j, (i+1)%dim_i,j] = 1.
                self.J_Mat[i,j, i,(j+1)%dim_i] = 1.

    def __init__(self):
        super(IsingHamiltonian, self).__init__(self._J)
        self.J_Mat = None

# Allow for arbitrary, random interaction between any two particles.
class SpinGlassHamiltonian(AbstractSpinGlassHamiltonian):
    # Simple lookup on pre-computed
    def _J(self, size):
        if self.J_Mat is None:
            dim_i, dim_j = size
            # Mode where we only allow interactions of {-1, +1}
            if self.categorical:
                self._dist = torch.distributions.binomial.Binomial(1, torch.tensor(.5))
                # Indecies l,m span the same ranges as (i, j) respectively.
                self.J_Mat = self._dist.sample((dim_i, dim_j, dim_i, dim_j))
                # Binomial outputs {0, 1}, but we need {-1, 1}. Replace 0 by -1.
                self.J_Mat[self.J_Mat==0] = -1
                self.J_Mat = self.J_Mat.float()
            # Otherwise grab interactions from a normal distribution
            else:
                mean,std = torch.tensor(0.0), torch.tensor(1.0)
                self._dist = torch.distributions.normal.Normal(mean, std)
                # Indecies l,m span the same ranges as (i, j) respectively.
                self.J_Mat = self._dist.sample((dim_i, dim_j, dim_i, dim_j))

    def __init__(self, categorical=False):
        super(SpinGlassHamiltonian, self).__init__(self._J)
        self.categorical = categorical
        # Generate J on first use. It's values don't really matter, as long as they are 
        # the same between timesteps. Defering generation to the call of _J means we don't
        # have to pre-compute the size of this thing in the command line.
        self.J_Mat = None


