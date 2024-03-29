import random
import math

class RBUF:
    
	def __init__(self, max_size=10000):
		self.cases = []
		self.max_size = max_size

	def add_case(self, feature, target):
		self.cases.append([feature, target])
		if len(self.cases) == self.max_size:
			del self.cases[0]

	def get_mini_batch(self, size):
		return random.sample(self.cases, int(size*len(self.cases)))
	
	def get_state_and_target_batch(self, size):
		num_cases = len(self.cases)
		if size > num_cases:
			size = num_cases
		batch = random.sample(self.cases, size)
		states = []
		targets = []

		for case in batch:
			states.append(case[0])
			targets.append(case[1])

		return states, targets