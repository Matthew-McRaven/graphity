import itertools
import functools

import torch
import torch.distributions
import numpy as np
    
@functools.lru_cache(1000)
def contribution(spins, J):
    local_spins = spins.clone()

    contribution = torch.zeros(local_spins.shape, dtype=torch.float32)
    # Unpack size of i,j dimensions, respectively
    dim_i, dim_j = local_spins.shape

    # Iterate over cartesian product of all index pairs.
    for (i,j) in itertools.product(range(dim_i), range(dim_j)):
        # Vectorize computation by computing all (l,m) pairs at once.
        # Scalar * 2d tensor * 2d tensor, summed.
        contribution[i,j] = (local_spins[i,j] * J[i,j] * local_spins).double().sum()
    return contribution

##########################
# Spin Glass Hamiltonian #
##########################
class AbstractSpinGlassHamiltonian():
    decomposable = True
    # "J" encapsulates the interaction between any two particles.
    def __init__(self, J):
        self.J = J


    def compute_glass(self, spins, J):
        contribs = contribution(spins, J)      
        energy = contribs.sum()

        # Energy is negative (Dr. Betre 20200105), but we need +'ve number to take logs.
        # And we need a -'ve number to minimize (for gradient descent).
        # So, in order to turn this number back into a "real" energy, compute
        #   -e^|energy|
        energy = self.normalize(energy)
        return energy, contribs

    def normalize(self, energy):
        ret =  -energy
        assert not torch.isinf(ret).any()

        return ret

    def fast_toggle(self, spins, contribution, site):
        self.J(spins.shape)
        # Unpack size of i,j dimensions, respectively
        changed_i, changed_j, old_site_value = site
        #print(f"Changed ({changed_i},{changed_j}) from {old_site_value} to {spins[changed_i, changed_j]}")
        # Only place each contribution can change is where it interacted with the changed site.
        # To compute new energy, subtract out the old contribution for each location, and add in the new contribution.
        contribution += spins * self.J_Mat[:,:, changed_i,changed_j] * (spins[changed_i,changed_j]-old_site_value)
        # However, the contribution for the change site is entirely hard, so let's just re-compute from scratch.
        contribution[changed_i, changed_j] = (spins[changed_i, changed_j] * self.J_Mat[changed_i, changed_j] * spins).double().sum()
        return contribution

    def contribution(self, spins):
        self.J(spins.shape)
        return contribution(spins, self.J_Mat)

    def __call__(self, spins, prev_contribs=None, changed_sites=None):
        # If we have no cached information from the previous timestep, we must perform full recompute
        if prev_contribs is None or changed_sites is None:
            self.J(spins.shape)
            energy, contribs =  self.compute_glass(spins, self.J_Mat)
            return energy, contribs

        # Otherwise we know the old value and what changed.
        # We can update the equation piecewise.
        else:
            contribs = prev_contribs
            #real, _ = self.compute_glass(spins, self.J_Mat)
            for site in changed_sites:
                contribs = self.fast_toggle(spins, contribs, site)
            energy = contribs.sum()
            
            
            
            # Energy is negative (Dr. Betre 20200105), but we need +'ve number to take logs.
            # And we need a -'ve number to minimize (for gradient descent).
            # So, in order to turn this number back into a "real" energy, compute
            #   -e^|energy|
            energy = self.normalize(energy)

           # p1 = imat.sum()/(imat.shape[0]*imat.shape[1]) 
            #ent = -(p1 * p1.log() + (1-p1)*(1-p1).log())
            #print(ent)
            
            #print()
            #assert (_ == contribs).all()
            #assert abs(real - energy) < 1
            return energy, contribs

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


