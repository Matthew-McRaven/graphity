import itertools
import torch
import graphity.site_strategy
import graphity.environment.lattice.reward
def test_mask_upper():
	tensor = torch.full((4,4), -1.0)
	tensor = graphity.site_strategy.mask_upper(tensor)
	for (i,j) in itertools.product(range(4), range(4)):
		if i < j: assert not torch.isneginf(tensor[j,i])
		else: assert torch.isneginf(tensor[j,i])
def test_score_1():
	tensor = torch.full((4,4), 1.0)
	H = graphity.environment.lattice.IsingHamiltonian()


	score_fn = graphity.site_strategy.Score1Neighbors(H)
	score = score_fn(tensor)

	# Flip the spin of the top left element.
	tensor[0,0]=-1.0
	score = score_fn(tensor)

	# Magic math from betre tells us our grad should be as follows.
	match = torch.tensor([[-8.,  4.,  8.,  4.],
        [ 4.,  8.,  8.,  8.],
        [ 8.,  8.,  8.,  8.],
        [ 4.,  8.,  8.,  8.]])
	assert (score == match).all()

def test_vanilla_ils():
	tensor = torch.full((4,4), 1.0)
	H = graphity.environment.lattice.IsingHamiltonian()
	score_fn = graphity.site_strategy.Score1Neighbors(H)
	ss = graphity.site_strategy.VanillaILS(score_fn)
	# Flip the spin of the top left element.
	tensor[0,0]=-1.0
	action, _ = ss(tensor)
	assert (action == torch.tensor([0, 0])).all()

def test_softmax_ils():
	tensor = torch.full((4,4), 1.0)
	H = graphity.environment.lattice.IsingHamiltonian()
	score_fn = graphity.site_strategy.Score1Neighbors(H)
	ss = graphity.site_strategy.SoftmaxILS(score_fn)
	# Flip the spin of the top left element.
	tensor[0,0] = -1.0
	action, _ = ss(tensor)

def test_beta_ils(): 
	tensor = torch.full((4,4), 1.0)
	H = graphity.environment.lattice.IsingHamiltonian()
	score_fn = graphity.site_strategy.Score1Neighbors(H)
	ss = graphity.site_strategy.BetaILS(score_fn)
	# Flip the spin of the top left element.
	tensor[0,0] = -1.0
	action, _ = ss(tensor)

def test_random_search():
	tensor = torch.full((4,4), 1.0)
	H = graphity.environment.lattice.IsingHamiltonian()
	ss = graphity.site_strategy.RandomSearch()
	# Flip the spin of the top left element.
	tensor[0,0] = -1.0
	action, _ = ss(tensor)