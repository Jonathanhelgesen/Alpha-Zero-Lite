import random
import math

class RBUF:
    
	def __init__(self):
		self.cases = []

	def add_case(self, case):
		self.cases.append(case)

	def add_cases(self, cases):
		self.cases.extend(cases)

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


	def get_weighted_mini_batch(lst, n):
		weights = [1.0 / math.sqrt(i+1) for i in range(len(lst))]
		weighted_lst = list(zip(lst, weights))
		chosen = random.choices(weighted_lst, weights=[w for e, w in weighted_lst], k=n)
		return [e for e, w in chosen]
