import itertools
import torch
import numpy as np


# Set the upper diagonal to -inf.
def mask_upper(lattice):
    # Prevent NN from toggling diagonal and duplicate edges in upper tril.
    upper_mask = torch.full(lattice.shape, -float('inf')).triu(0)
    return lattice + upper_mask

# Compute the the score of each of the 1-neighbors of the graph.
# That is, compute the Δe of all lattices with one only one spin different than the base lattics.
class Score1Neighbors:
    def __init__(self, H):
        self.H = H
    def __call__(self, spins):
        local_spins = spins.clone().detach()
        score = torch.zeros(*local_spins.shape)
        dims = [range(x) for x in local_spins.shape]
        contrib = self.H.contribution(local_spins)
        current_energy = contrib.sum()
        for dim in itertools.product(*dims):
            dim = tuple(dim)
            site_val = local_spins[dim]
            local_spins[dim] *= -1
            new_energy, new_contrib = self.H(local_spins, prev_contribs=contrib, changed_sites=[(dim, site_val)])
            local_spins[dim] *= -1
            score[dim] =  new_energy - current_energy
        return score

class IteratedLocalSearch:
    def __init__(self, score_fn=None, mask_triu=True):
        assert score_fn
        self.score_fn = score_fn
        self.mask_triu = mask_triu
    def __call__(self, lattice):
        scored_neighbors = self.score_fn(lattice)
        return mask_upper(scored_neighbors) if self.mask_triu else scored_neighbors

class VanillaILS(IteratedLocalSearch):
    def __init__(self, score_fn=None, mask_triu=False):
        super(VanillaILS, self).__init__(score_fn=score_fn, mask_triu=mask_triu)

    def __call__(self, lattice):
        scored_neighbors = super(VanillaILS, self).__call__(lattice)

        # Pick top transition than most minimizes energy.
        _, index = torch.topk(scored_neighbors.view(-1), 1, largest=False)
        picked = np.unravel_index(index.item(), scored_neighbors.shape)
        site = torch.tensor(picked[:]).view(-1).to(lattice.device)
        return site, torch.zeros((1,), device=lattice.device)

class SoftmaxILS(IteratedLocalSearch):
    def __init__(self, score_fn=None, mask_triu=False): 
        super(SoftmaxILS, self).__init__(score_fn, mask_triu)

    def __call__(self, lattice):
        scored_neighbors = super(SoftmaxILS, self).__call__(lattice)
        probs = torch.softmax(scored_neighbors.view(-1), 0).view(-1)

        # Pick a transition according to its softmax probability.
        dist = torch.distributions.categorical.Categorical(probs)
        index = dist.sample((1,))
        picked = np.unravel_index(index.item(), scored_neighbors.shape)
        site = torch.tensor(picked[:]).view(-1).to(lattice.device)

        return site, dist.log_prob(index)


class BetaILS(IteratedLocalSearch):
    def __init__(self, score_fn=None, alpha=.5, beta=2, mask_triu=False):
        super(BetaILS, self).__init__(score_fn, mask_triu)
        self.dist = torch.distributions.beta.Beta(alpha, beta)

    def __call__(self, lattice):
        scored_neighbors = self.score_fn(lattice)
        min, max = torch.min(scored_neighbors.view(-1)), torch.max(scored_neighbors.view(-1))
        scored_neighbors = (scored_neighbors - min)/ (max - min)
        scored_neighbors = mask_upper(scored_neighbors) if self.mask_triu else scored_neighbors
        # Pick a transiiton whose value is closest to the sampled value.
        value = self.dist.sample((1,))
        index = torch.argmin((scored_neighbors-value).abs())
        picked = np.unravel_index(index.item(), scored_neighbors.shape)
        site = torch.tensor(picked[:]).view(-1).to(lattice.device)
        return site, self.dist.log_prob(value)
