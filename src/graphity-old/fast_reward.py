import itertools
import functools

import torch
import torch.distributions
import numpy as np
    
@functools.lru_cache(1000)
def contribution(spins, J):
    local_spins = spins.clone()
    # Spins ∈ {-1, 1}, but adjacency's are ∈ {0, 1}
    local_spins[local_spins==0] = -1
    contribution = torch.zeros(local_spins.shape, dtype=torch.float32)
    # Unpack size of i,j dimensions, respectively
    dim_i, dim_j = local_spins.shape

	# TODO: This is what we need to work on!
	
    # Iterate over cartesian product of all index pairs.
    # We know a priori what the spin_ij, and spin_lm are, but we need to be told by
    # our subclass how to compute J, so defer to self.J
    for (i,j) in itertools.product(range(dim_i), range(dim_j)):
        # Vectorize computation by computing all (l,m) pairs at once.
        # Scalar * 2d tensor * 2d tensor, summed.
        contribution[i,j] = (local_spins[i,j] * local_spins * J[i,j]).double().sum()
    return contribution

##########################
# Spin Glass Hamiltonian #
##########################
class DecomposedA2DHamiltonian():
    decomposable = True
    # "J" encapsulates the interaction between any two particles.
    def __init__(self, J):
        self.J = J


    def compute_glass(self, spins):
        contribs = contribution(spins, self.L_Mat)      
        energy = contribs.sum()

        energy = self.normalize(energy)
        return energy, contribs

    def normalize(self, energy):
        assert not torch.isinf(energy).any()

        return energy

    def fast_toggle(self, spins, contribution, site):
        self.J(spins.shape)
        # Unpack size of i,j dimensions, respectively
        changed_i, changed_j, old_site_value = site
        # TODO: Must update based on old site values
        # This likely involves change the "2" term out fron.
        delta = spins[changed_i, changed_j] - old_site_value 
        #print(contribution)
        # TODO: I am really suspicious that -spins is wrong.
        contribution -= 2 * -spins * self.J_Mat[:,:, changed_i, changed_j]
        contribution[changed_i, changed_j] *= -1
        return contribution

    def contribution(self, spins):
        self.J(spins.shape)
        return contribution(spins, self.J_Mat)

    def __call__(self, spins, prev_contribs=None, changed_sites=None):
        # If we have no cached information from the previous timestep, we must perform full recompute
        if prev_contribs is None or changed_sites is None:
            self.L(spins.shape)
            return self.compute_glass(spins)

        # Otherwise we know the old value and what changed.
        # We can update the equation piecewise.
        else:
            contribs = prev_contribs
            for site in changed_sites:
                contribs = self.fast_toggle(spins, contribs, site)
            energy = contribs.sum()
            # Energy is negative (Dr. Betre 20200105), but we need +'ve number to take logs.
            # And we need a -'ve number to minimize (for gradient descent).
            # So, in order to turn this number back into a "real" energy, compute
            #   -e^|energy|
            energy = self.normalize(energy)
            return energy, contribs

# Nearest-neighbor interactions only.
class A2D(DecomposedA2DHamiltonian):
    # Simple lookup on pre-computed
    def _L(self, size):
        assert size[0] == size[1]
        if self.L_Mat is None:
            dim_i, dim_j = size
            self.L_Mat = torch.ones_like((2*self.d),(dim_i, dim_j), dtype=torch.float32)
            for i in range(dim_i):
                self.L_Mat[i,i] = ??

    def __init__(self, d):
        super(A2D, self).__init__(self._J)
        self.L_Mat = None
        self.d = d


