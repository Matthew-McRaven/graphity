import itertools

import torch

import graphity.strategy.anneal

##################################################################
# Test that temperature modification schedules work as predicted #
##################################################################

# Test that the constant temperature schedule never changes.
def test_const_temp():
	start_beta = beta = 5
	strat = graphity.strategy.anneal.ConstBeta(beta)
	for i in range(10):
		new_beta, _ = strat(beta, 100)
		assert new_beta == start_beta
		beta = new_beta


# Simulated annealing with a round length of 1.
def test_simulated_annealing_length_1():
	alpha = .5
	strat = graphity.strategy.anneal.SimulatedAnnealing(alpha, 1, .005)
	expected = [0.005, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56]
	for i in range(10):
		beta, _ = strat(100)
		strat.step()
		assert beta == expected[i]

# Simulated annealing with a round length of 3.
def test_simulated_annealing_length_3():
	alpha = .5
	strat = graphity.strategy.anneal.SimulatedAnnealing(alpha, 3, .005)
	expected = [0.005, 0.005, 0.005, 0.01, 0.01, 0.01, 0.02, 0.02, 0.02, 0.04, 0.04]
	for i in range(10):
		beta, _ = strat(100)
		strat.step()
		assert beta == expected[i]