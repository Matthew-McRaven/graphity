
import numpy as np
from numpy.random import Generator, PCG64

# Class representing a weighted task distribution.
# You can sample a number of tasks from this with replacements
class TaskDistribution:
	def __init__(self, seed=None):
		self._tasks = []
		self._task_weights = []
		self._rng = Generator(PCG64(seed))

	def add_task(self, task, weight=1):
		self._tasks.append(task)
		self._task_weights.append(weight)

	def sample(self, count):
		# Must convert weights to proabbilities.
		mean = sum(self._task_weights)
		task_probabilities = list(map(lambda x: x/mean, self._task_weights))
		# Task definitions are different than task objects.
		# If they were the same, and we sampled on tasks directly,
		# it was possible to return the same task object twice.
		# This lead to accidentally overwriting replay memory.
		definitions = self._rng.choice(self._tasks, size=(count,), replace=True, p=task_probabilities)
		return [definition.instance() for definition in definitions]


