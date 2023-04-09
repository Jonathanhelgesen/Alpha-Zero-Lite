from abc import ABC, abstractmethod


class Game(ABC):
	"""Abstract class for two-player games"""
    
	@abstractmethod
	def get_state(self):
		pass


	@abstractmethod
	def execute_move(self, move, player):
		pass


	@abstractmethod
	def get_valid_moves(state):
		pass


	@abstractmethod
	def is_move_valid(state, move):
		pass


	@abstractmethod
	def get_game_status(self, player):
		pass


	def get_opponent(player):
		"""1 represents current player, -1 is the opponent"""
		return -player
	

	@abstractmethod
	def undo_move(self, move):
		pass


	@abstractmethod
	def check_status(self, move):
		pass


	@abstractmethod
	def get_move_count(self):
		pass