import itertools
import functools

import torch
import torch.distributions
import numpy as np
    

def contribution(spins, J):
    """
    Computes the contribution of each site to the overall energy of a glass.
    This allows faster re-computation on toggles, as you only have to update sites who interact with the changed value.
    """
    contribution = torch.zeros(spins.shape, dtype=torch.float32)
    # Unpack size of i,j dimensions, respectively
    dims = [range(dim) for dim in spins.shape]

    # Iterate over cartesian product of all index pairs.
    for site in itertools.product(*dims):
        site = tuple(site)
        # Vectorize computation by computing all (l,m) pairs at once.
        # Scalar * 2d tensor * 2d tensor, summed.
        contribution[site] = -.5*(spins[site] * J[site] * spins).double().sum()
    return contribution

##########################
# Spin Glass Hamiltonian #
##########################
class AbstractSpinGlassHamiltonian():
    """
    Abstract base class for any spin glass Hamiltonians.
    Reduces code reuse by implementing methods shared between other spin glass Hamiltonians.

    Deriving classes must specify a class attribute, J_Mat, a 4-d tensor that records the strength of interaction between
    site (i,j) and site (k,l) by the equation `self.J_Mat[i, j, k, l]`. This tensor is generated on first use by calling
    self.J(some_glass_here), so that no a-priori knowledge of glass shape is needed. Glass shapes must not change at runtime.
    """
    # "J" encapsulates the interaction between any two particles.
    def __init__(self, J):
        """
        :param J: A function of an lattice which will fill in self.J_Mat
        """
        self.J = J


    def compute_glass(self, spins, J):
        """
        Recompute the energy of a spin glass from scratch. Must match in signature with operator().

        :param spins: A tensor containing a spin glass.
        :param J: A 4-d tensor recording interaction between two sites.
        """
        contribs = contribution(spins, J)   
        energy = contribs.sum()

        return energy, contribs

    def fast_toggle(self, spins, contrib, site):
        """
        Using knowledge of which sites changed and what old contributions were, recompute only contributions that changed.
        This is much more efficient than calling recompute_glass(...) on every change.

        Extensive unit tests indicate that this method is exactly equivalent to a full recomputation.
        Additionally, at one point, I had a formula that proved this.
        In the future, I may add the proof here.

        :param spins: A tensor containing a spin glass. The desired change(s) has already been applied to this glass.
        :param contribs: A tensor containing the energy each site contributes to the total energy. The desired change(s)
        have not yet been applied to this tensor.
        :param site: A list of (index, old_value) pairs which contain the location and old value of all sites that have been toggled.
        """

        contribution = contrib.detach().clone()
        self.J(spins.shape)
        # Unpack size of i,j dimensions, respectively
        site_index, old_site_value = site
        slice_list = len(site_index) * [slice(0, None),]
        slice_list.extend([idx for idx in site_index])
        # Only place each contribution can change is where it interacted with the changed site.
        # To compute new energy, subtract out the old contribution for each location, and add in the new contribution.
        contribution -= spins * self.J_Mat[tuple(slice_list)] *old_site_value
        # However, the contribution for the change site is entirely hard, so let's just re-compute from scratch.
        site = tuple(site_index)
        contribution[site] = -.5*(spins[site] * self.J_Mat[site] * spins).double().sum()
        return contribution

    def contribution(self, spins):
        """
        Determine the contribution of each site to the total energy of the spin glass.

        :param spins: A tensor containing a spin glass. 
        """
        self.J(spins.shape)
        return contribution(spins, self.J_Mat)

    def __call__(self, spins, prev_contribs=None, changed_sites=None):
        """
        Compute the energy of a spin glass.

        If either prev_contrib or changed_sites are `None`, then the spin glass will be entirely re-computed.
        If both contain relevant data, a faster computation can take place.

        :param spins: A tensor containing a spin glass. The desired change(s) has already been applied to this glass.
        :param contribs: A tensor containing the energy each site contributes to the total energy. The desired change(s)
        have not yet been applied to this tensor. If None, a full recalculation will be performed instead of a fast partial computation.
        Defaults to None.
        :param site: A list of (index, old_value) pairs which contain the location and old value of all sites that have been toggled.
        If None, a full recalculation will be performed instead of a fast partial computation.
        Defaults to None.
        """
        # If we have no cached information from the previous timestep, we must perform full recompute
        if prev_contribs is None or changed_sites is None:
            self.J(spins.shape)
            energy, contribs =  self.compute_glass(spins, self.J_Mat)
            return energy, contribs

        # Otherwise we know the old value and what changed.
        # We can update the equation piecewise.
        else:
            contribs = prev_contribs
            for site in changed_sites:
                contribs = self.fast_toggle(spins, contribs, site)
            energy = contribs.sum()
            
            return energy, contribs

# Nearest-neighbor interactions only.
class IsingHamiltonian(AbstractSpinGlassHamiltonian):
    """
    A Hamiltonian for spin glasses with only nearest-neighbor interactions. 
    These interactions have unit strength.
    All other interactions are 0.
    """

    def _J(self, size):
        """
        Function passed as J to AbstractSpinGlassHamiltonian.__init__().
        Initializes J_Mat to the correct coupling tensor.
        """
        # TODO: Extend for arbitrary dimensionality.
        if self.J_Mat is None:
            if len(size) == 1:
                dim_i, = size
                self.J_Mat = torch.zeros((dim_i, dim_i), dtype=torch.float32)
                # Fill in the "adjacency" matrix for the ising model.
                for i in range(dim_i):
                    self.J_Mat[i, i-1] = 1.
                    self.J_Mat[i, (i+1)%dim_i] = 1.
            elif len(size) == 2:
                dim_i, dim_j = size
                self.J_Mat = torch.zeros((dim_i, dim_j, dim_i, dim_j), dtype=torch.float32)
                # Fill in the "adjacency" matrix for the ising model.
                for (i,j) in itertools.product(range(dim_i), range(dim_j)):
                    self.J_Mat[i,j, i-1,j] = 1.
                    self.J_Mat[i,j, i,j-1] = 1.
                    self.J_Mat[i,j, (i+1)%dim_i,j] = 1.
                    self.J_Mat[i,j, i,(j+1)%dim_j] = 1.
            elif len(size) == 3:
                dim_i, dim_j, dim_k = size
                self.J_Mat = torch.zeros((dim_i, dim_j, dim_k, dim_i, dim_j, dim_k), dtype=torch.float32)
                # Fill in the "adjacency" matrix for the ising model.
                for (i,j,k) in itertools.product(range(dim_i), range(dim_j), range(dim_k)):
                    self.J_Mat[i,j,k, i-1,j,k] = 1.
                    self.J_Mat[i,j,k, i,j-1,k] = 1.
                    self.J_Mat[i,j,k, i,j,k-1] = 1.
                    self.J_Mat[i,j,k, (i+1)%dim_i,j,k] = 1.
                    self.J_Mat[i,j,k, i,(j+1)%dim_i,k] = 1
                    self.J_Mat[i,j,k, i,j,(k+1)%dim_i] = 1
            else:
                raise NotImplemented("TODO!")

    def __init__(self):
        super(IsingHamiltonian, self).__init__(self._J)
        self.J_Mat = None

class ConstInfiniteRangeHamiltonian(AbstractSpinGlassHamiltonian):
    """
    A Hamiltonian for spin glasses with infinite-range interactions.
    All interactions have the same strength.
    """
    def _J(self, size):
        """
        Function passed as J to AbstractSpinGlassHamiltonian.__init__().
        Initializes J_Mat to the correct coupling tensor.
        """
        if self.J_Mat is None:
            dims = size+size
            self.J_Mat = torch.full(dims, self.constant, dtype=torch.float32)

    def __init__(self, constant=1.0):
        super(ConstInfiniteRangeHamiltonian, self).__init__(self._J)
        self.constant = constant
        self.J_Mat = None

# Allow for arbitrary, random interaction between any two particles.
class SpinGlassHamiltonian(AbstractSpinGlassHamiltonian):
    """
    A Hamiltonian for spin glasses with infinite-range interactions.
    All interactions have random strength.

    Strengths can either be drawn from a categorical distribution (i.e., {-1, +1}) or a normal distribution (i.e., N(0,1)).
    """
    def _J(self, size):
        """
        Function passed as J to AbstractSpinGlassHamiltonian.__init__().
        Initializes J_Mat to a random coupling matrix.
        """
        if self.J_Mat is None:
            size = tuple(2*size)
            # Mode where we only allow interactions of {-1, +1}
            if self.categorical:
                self._dist = torch.distributions.binomial.Binomial(1, torch.tensor(.5))
                # Indecies l,m span the same ranges as (i, j) respectively.
                self.J_Mat = self._dist.sample((*size,))
                # Binomial outputs {0, 1}, but we need {-1, 1}. Replace 0 by -1.
                self.J_Mat[self.J_Mat==0] = -1
                self.J_Mat = self.J_Mat.float()
            # Otherwise grab interactions from a normal distribution
            else:
                mean,std = torch.tensor(0.0), torch.tensor(1.0)
                self._dist = torch.distributions.normal.Normal(mean, std)
                # Indecies l,m span the same ranges as (i, j) respectively.
                self.J_Mat = self._dist.sample((*size,))

    def __init__(self, categorical=False):
        """
        :param categorical: If truthy, draw interactions from {+1, -1}. Otherwise, draw interactions from N(0, 1) 
        """
        super(SpinGlassHamiltonian, self).__init__(self._J)
        self.categorical = categorical
        # Generate J on first use. It's values don't really matter, as long as they are 
        # the same between timesteps.
        self.J_Mat = None


