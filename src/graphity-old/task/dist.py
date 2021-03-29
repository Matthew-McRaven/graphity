import numpy as np
from numpy.random import Generator, PCG64

# Task distribution from which every task is sample each epoch.
class TaskDistribution:
    def __init__(self):
        self._tasks = []

    def add_task(self, task):
        self._tasks.append(task)

    def gather(self):
        return [definition.instance() for definition in self._tasks]
