import itertools

import torch

import graphity.environment.graph.reward
import graphity.strategy.site


###########################################################
# Test that we can mask out the upper portion of a matrix #
###########################################################
def test_mask_upper():
	tensor = torch.full((4,4), -1.0)
	tensor = graphity.strategy.site.mask_upper(tensor)
	for (i,j) in itertools.product(range(4), range(4)):
		if i < j: assert not torch.isneginf(tensor[j,i])
		else: assert torch.isneginf(tensor[j,i])

#########################################################
# Test that site selection strategies work as predicted #
#########################################################

# Evaluate all future states which are within one toggle of the current state.
def test_score_1():
	tensor = torch.full((4,4), 1.0)
	H = graphity.environment.graph.ASquaredD(2, keep_diag=False)

	score_fn = graphity.strategy.site.Score1Neighbors(H)
	score = score_fn(tensor)
	print(score)

	# Flip the spin of the top left element.
	tensor[0,0]=-1.0
	score = score_fn(tensor)
	print(score)
	# Magic math from betre tells us our grad should be as follows.
	match = torch.tensor([[  12., 0., 0., 0.],
        [0., -4., 0., 0.],
        [0., 0., -4.,0.],
        [0., 0., 0., -4.]])
	assert (score == match).all()

# Pick the transition within 1 toggle that maximally minimizes the energy
def test_vanilla_ils():
	tensor = torch.full((4,4), 1.0)
	H = graphity.environment.graph.ASquaredD(2)
	score_fn = graphity.strategy.site.Score1Neighbors(H)
	ss = graphity.strategy.site.VanillaILS(score_fn)
	# Flip the spin of the top left element.
	tensor[0,0]=-1.0
	action, _ = ss(tensor)
	#assert (action == torch.tensor([0, 0])).all()
	# TODO: I do not know what the result should be.

# Treat the 1-neighbor transition scores as a probability distribution.
# Randomly sample a site from this probability distribution.
def test_softmax_ils():
	tensor = torch.full((4,4), 1.0)
	H = graphity.environment.graph.ASquaredD(2)
	score_fn = graphity.strategy.site.Score1Neighbors(H)
	ss = graphity.strategy.site.SoftmaxILS(score_fn)
	# Flip the spin of the top left element.
	tensor[0,0] = -1.0
	action, _ = ss(tensor)
	# Can't test what action is performed, because it is non-deterministic.

# Treat the 1-neighbor transition scores as a probability distribution.
# This distribution is the beta distribution, which heabily biases choices towards the "best" transition.
# Randomly sample a site from this probability distribution.
def test_beta_ils(): 
	tensor = torch.full((4,4), 1.0)
	H = graphity.environment.graph.ASquaredD(2)
	score_fn = graphity.strategy.site.Score1Neighbors(H)
	ss = graphity.strategy.site.BetaILS(score_fn)
	# Flip the spin of the top left element.
	tensor[0,0] = -1.0
	action, _ = ss(tensor)
	# Can't test what action is performed, because it is non-deterministic.

# Just randomly pick a site.
# No clever thoughts on this one.
def test_random_search():
	tensor = torch.full((4,4), 1.0)
	H = graphity.environment.graph.ASquaredD(2)
	ss = graphity.strategy.site.RandomSearch()
	# Flip the spin of the top left element.
	tensor[0,0] = -1.0
	action, _ = ss(tensor)
	# Can't test what action is performed, because it is non-deterministic.