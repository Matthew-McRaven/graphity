from collections import Counter
import platform
import time
import ray
import os
from threading import Thread, Lock

ray.init(address='auto')

def f(x):
    time.sleep(2)
    return x + 1
class Task:
	def __init__(self):
		self.epoch = 0
		self.data = None
		self.energy = 0

@ray.remote
class controller:
	def __init__(self, task_count):
		self.task_count = 100
		self.available_tasks = [(x, Task()) for x in range(self.task_count)]
		self.mut = Lock()
		self.epoch = 0

	def get_work(self):
		self.mut.acquire()
		rval = None
		try:
			rval = self.available_tasks.pop(0)
		finally:
			self.mut.release()
		return rval
	def cont(self):
		return self.epoch < 5
	def return_work(self, index, task):
		self.mut.acquire()
		if task.epoch > self.epoch:
			self.epoch = task.epoch
		try:
			rval = self.available_tasks.append((index, task))
		finally:
			self.mut.release()

@ray.remote
class worker:
	def __init__(self, ctrl):
		self.ctrl = ctrl
	def work_cycle(self):
		index, task = ray.get(ctrl.get_work.remote())
		print(f"Working with {index} with epoch {task.epoch}")
		time.sleep(.1)
		task.epoch += 1
		ctrl.return_work.remote(index, task)
	def run(self):
		while ray.get(self.ctrl.cont.remote()):
			self.work_cycle()
			
ctrl = controller.remote(100)
# Check that objects can be transferred from each node to each other node.
workers = [worker.remote(ctrl) for _ in range(10)]
ray.get([worker.run.remote() for worker in workers])
