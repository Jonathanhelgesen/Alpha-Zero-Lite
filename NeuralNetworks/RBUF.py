import random
import math

class RBUF:
    
	def __init__(self):
		self.cases = []

	def add_case(self, case):
		self.cases.append(case)

	def get_mini_batch(self, size):
		return random.sample(self.cases, size)
	
	def get_weighted_mini_batch(lst, n):
		weights = [1.0 / math.sqrt(i+1) for i in range(len(lst))]
		weighted_lst = list(zip(lst, weights))
		chosen = random.choices(weighted_lst, weights=[w for e, w in weighted_lst], k=n)
		return [e for e, w in chosen]
