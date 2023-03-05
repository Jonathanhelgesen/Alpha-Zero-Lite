import sys
sys.path.append("..")
from Game import Game

class Nim(Game):
	"""Simple implementation of the game Nim"""

	def __init__(self, N, K, starting_player):
		self.stones_left = N
		self.K = K
		self.starting_player = starting_player
		self.turn = 0


	def get_game_status(self, player):
		if  self.stones_left <= 0:
			# Player that moved lost
			return -player
		else:
			return 0


	def is_move_valid(self, move):
		return 0 < move <= self.K
		

	def make_move(self, k):
		player = self.starting_player if self.turn % 2 == 0 else -self.starting_player

		if self.is_move_valid(k):
			self.stones_left -= k
		else:
			raise ValueError(f'Cannot remove more than {self.K} stones')
		
		turn += 1
		status = self.get_game_status(player)

		return status
	

	def get_state(self):
		return self.stones_left, self.K
	

	def get_valid_moves(self):
		return range(self.K + 1)