import random


class DecentNimPlayer():
	"""Nim player that always try leave one stone on the board
	if possible. Else, take a random number of stones"""

	def __init__(self, K):
		self.K = K

	def make_move(self, stones_left):
		if stones_left <= self.K + 1:
			return stones_left - 1
		else:
			return random.randint(1, self.K)