import random
import math

class RBUF2:
    
	def __init__(self, max_size=1024):
		self.cases = []
		self.max_size = max_size

	def add_case(self, board, player, target):
		self.cases.append([[board, player], target])
		num_cases = len(self.cases)
		print(f'Number of cases in RBUF: {num_cases}')
		if num_cases == self.max_size:
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
			#self.cases.remove(case)
			states.append(case[0])
			targets.append(case[1])

		return states, targets