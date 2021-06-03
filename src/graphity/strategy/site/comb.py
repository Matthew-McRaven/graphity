import itertools
import torch
import numpy as np


# Set the upper diagonal to -inf.
def mask_upper(lattice):
    # Prevent toggling in upper tril.
    upper_mask = torch.full(lattice.shape, float('inf')).triu(0)
    return lattice + upper_mask


class Score1Neighbors:
    """
    Compute the the score of each of the 1-neighbors of the graph.
    That is, compute the Δe of all lattices with one only one spin different than the base lattices.
    The i'th, j'th, (maybe k'th) index corresponds to the Δe of the graph formed by taking the original graph and flipping it's (i, j, (k?))'th spin.
    """
    def __init__(self, H):
        ":param H: Hamiltonian used to score states' energies."
        self.H = H
    def __call__(self, spins):
        # Make a copy so that we don't accidentally destory the source data or affect its grads.
        local_spins = spins.clone().detach()
        # Create a tensor to hold all the results of our 1-neighbor's toggles.
        score = torch.zeros(*local_spins.shape)
        dims = [range(x) for x in local_spins.shape]
        # Pre-compute the contriputions, to speed up our Hamiltonian computation,
        contrib = self.H.contribution(local_spins)
        current_energy = contrib.sum()
        for dim in itertools.product(*dims):
            # Must be a tuple. Being a list/iterable breaks indexing.
            dim = tuple(dim)
            # Must remember original site value to take advantage of fast H computation
            site_val = local_spins[dim]
            local_spins[dim] *= -1
            new_energy, _ = self.H(local_spins, prev_contribs=contrib, changed_sites=[(dim, site_val)])
            local_spins[dim] *= -1
            # Store Δe.
            score[dim] =  new_energy - current_energy
        return score

class IteratedLocalSearch:
    """
    A Helper class which performs an iterated local search over a lattice's 1-neighborhood for the best possible
    transition,
    """
    def __init__(self, score_fn=None, mask_triu=True):
        """
        :param score_fn: A function (Hamiltonian) used to assign energies to states.
        :param mask_triu: If true, never pick a sample from the upper triangle.
        """
        assert score_fn
        self.score_fn = score_fn
        self.mask_triu = mask_triu

    def __call__(self, lattice):
        scored_neighbors = self.score_fn(lattice)
        return mask_upper(scored_neighbors) if self.mask_triu else scored_neighbors

class VanillaILS(IteratedLocalSearch):
    """
    Examine the 1-neighbors of the current lattice (that is, all lattices within 1 spin flip of this lattice), and
    return the site which most minimizes the energy of the system.
    """
    def __init__(self, score_fn=None, mask_triu=False):
        """
        :param score_fn: A function (Hamiltonian) used to assign energies to states.
        :param mask_triu: If true, never pick a sample from the upper triangle.
        """
        super(VanillaILS, self).__init__(score_fn=score_fn, mask_triu=mask_triu)

    def __call__(self, lattice):
        scored_neighbors = super(VanillaILS, self).__call__(lattice)

        # Pick top transition than most minimizes energy.
        _, index = torch.topk(scored_neighbors.view(-1), 1, largest=False)
        picked = np.unravel_index(index.item(), scored_neighbors.shape)
        site = torch.tensor(picked[:]).view(-1).to(lattice.device)
        return site, torch.zeros((1,), device=lattice.device)

class SoftmaxILS(IteratedLocalSearch):
    """
    Examine the 1-neighbors of the current lattice (that is, all lattices within 1 spin flip of this lattice).
    Treat these energies as a distribution by feeding them through a softmax.
    Then, sample a site from this probability distribution.

    This gives high weights to large negative transitions, and small probabilities to positive transitions.
    However, due to its probabalistic nature, it is not as likely to become stuck as the Vanilla ILS algorithm.
    """
    def __init__(self, score_fn=None, mask_triu=False): 
        """
        :param score_fn: A function (Hamiltonian) used to assign energies to states.
        :param mask_triu: If true, never pick a sample from the upper triangle.
        """
        super(SoftmaxILS, self).__init__(score_fn, mask_triu)

    def __call__(self, lattice):
        scored_neighbors = super(SoftmaxILS, self).__call__(lattice)
        # Softmax expects negative numbers to be small probabilites, but we want negative energies to be more likely.
        # Thereforem swap our signs.
        probs = torch.softmax(-scored_neighbors.view(-1), 0).view(-1)

        # Pick a transition according to its softmax probability.
        dist = torch.distributions.categorical.Categorical(probs)
        index = dist.sample((1,))
        # And cleverly convert from a 1d index to a 2d index.
        picked = np.unravel_index(index.item(), scored_neighbors.shape)
        site = torch.tensor(picked[:]).view(-1).to(lattice.device)

        return site, dist.log_prob(index)


class BetaILS(IteratedLocalSearch):
    """
    Examine the 1-neighbors of the current lattice (that is, all lattices within 1 spin flip of this lattice).
    Treat these energies as samples from a beta distribution by remapping them to [0,1].
    Then, draw a sample from a beta distribution, and pick the remapped energy that is closest to the drawn value.

    This gives us a finer ability to control the shape of the transition distribution. 
    In practice, it appears to outperfom a Softmax prob distribution.
    """
    def __init__(self, score_fn=None, alpha=.5, beta=2, mask_triu=False):
        """
        :param score_fn: A function (Hamiltonian) used to assign energies to states.
        :param alpha: The alpha parameter of the underlying beta distribution.
        :param beta: The beta parameter of the underlying beta distribution.
        :param mask_triu: If true, never pick a sample from the upper triangle.
        """
        super(BetaILS, self).__init__(score_fn, mask_triu)
        self.dist = torch.distributions.beta.Beta(alpha, beta)

    def __call__(self, lattice):
        scored_neighbors = self.score_fn(lattice)
        min, max = torch.min(scored_neighbors.view(-1)), torch.max(scored_neighbors.view(-1))
        scored_neighbors = (scored_neighbors - min)/ (max - min)
        scored_neighbors = mask_upper(scored_neighbors) if self.mask_triu else scored_neighbors
        # Pick a transition whose value is closest to the sampled value.
        value = self.dist.sample((1,))
        index = torch.argmin((scored_neighbors-value).abs())
        picked = np.unravel_index(index.item(), scored_neighbors.shape)
        site = torch.tensor(picked[:]).view(-1).to(lattice.device)
        return site, self.dist.log_prob(value)
