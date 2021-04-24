import itertools
import torch
import graphity.strategy.site
import graphity.environment.graph.reward
def test_mask_upper():
	tensor = torch.full((4,4), -1.0)
	tensor = graphity.strategy.site.mask_upper(tensor)
	for (i,j) in itertools.product(range(4), range(4)):
		if i < j: assert not torch.isneginf(tensor[j,i])
		else: assert torch.isneginf(tensor[j,i])
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
	match = torch.tensor([[  0., -12., -12., -12.],
        [-12., -16., -12., -12.],
        [-12., -12., -16., -12.],
        [-12., -12., -12., -16.]])
	assert (score == match).all()

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

def test_softmax_ils():
	tensor = torch.full((4,4), 1.0)
	H = graphity.environment.graph.ASquaredD(2)
	score_fn = graphity.strategy.site.Score1Neighbors(H)
	ss = graphity.strategy.site.SoftmaxILS(score_fn)
	# Flip the spin of the top left element.
	tensor[0,0] = -1.0
	action, _ = ss(tensor)
	# Can't test what action is performed, because it is non-deterministic.

def test_beta_ils(): 
	tensor = torch.full((4,4), 1.0)
	H = graphity.environment.graph.ASquaredD(2)
	score_fn = graphity.strategy.site.Score1Neighbors(H)
	ss = graphity.strategy.site.BetaILS(score_fn)
	# Flip the spin of the top left element.
	tensor[0,0] = -1.0
	action, _ = ss(tensor)
	# Can't test what action is performed, because it is non-deterministic.

def test_random_search():
	tensor = torch.full((4,4), 1.0)
	H = graphity.environment.graph.ASquaredD(2)
	ss = graphity.strategy.site.RandomSearch()
	# Flip the spin of the top left element.
	tensor[0,0] = -1.0
	action, _ = ss(tensor)
	# Can't test what action is performed, because it is non-deterministic.