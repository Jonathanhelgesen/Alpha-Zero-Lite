from DisjointSet import DisjointSet
import sys
sys.path.append("..")
from Game import Game


class Hex(Game):

	def __init__(self, size):
        # TODO: Change 0, 1 and -1 to [0, 0], [1, 0] and [0, 1] for ANN
		self.size = size
		self.board = [[0] * size for _ in range(size)]
		self.cells = [(i, j) for i in range(size) for j in range(size)]

		self.left_node = (0, -1)
		self.right_node = (0, size)
		self.top_node = (-1, 0)
		self.bottom_node = (size, 0)

		self.ds_current_player = DisjointSet(self.cells + [self.left_node, self.right_node])
		self.ds_opponent = DisjointSet(self.cells + [self.top_node, self.bottom_node])

		for i in range(size):
			self.ds_current_player.join((i, 0), self.left_node)
			self.ds_current_player.join((i, size - 1), self.right_node)
			self.ds_opponent.join((0, i), self.top_node)
			self.ds_opponent.join((size - 1, i), self.bottom_node)


	def is_move_valid(self, move):
		r = move[0]
		c = move[1]
		if 0 <= r < self.size and 0 <= c < self.size:
			print(self.board)
			print(f'{r}_{c}')
			if self.board[r][c] == 0:
				return True
			else:
				return False
		else:
			return False
		

	def get_game_status(self, move, player):
		"""Returns the game status, where 1 is win, 0 is not finished and -1 is loss"""
		r = move[0]
		c = move[1]
		for r_neighbour, c_neighbour in [(r + 1, c), (r + 1, c - 1), (r, c + 1), 
										 (r, c - 1), (r - 1, c), (r - 1, c + 1)]:
			if 0 <= r_neighbour < self.size and 0 <= c_neighbour < self.size and self.board[r_neighbour][c_neighbour] == player:
				if player == 1:
					self.ds_current_player.join((r_neighbour, c_neighbour), (r, c))
				else:
					self.ds_opponent.join((r_neighbour, c_neighbour), (r, c))

		if self.ds_current_player.find(self.left_node) == self.ds_current_player.find(self.right_node):
			return 1
		elif self.ds_opponent.find(self.top_node) == self.ds_opponent.find(self.bottom_node):
			return -1
		else:
			# Game is not finished
			return 0


	def execute_move(self, move, player):
		"""Executes the given move, if allowed. Returns the status of the game"""
		r = move[0]
		c = move[1]
		if self.is_move_valid(move) and player in [1, -1]:
			self.board[r][c] = player
		else:
			raise ValueError('Move is not allowed')

		status = self.get_game_status(move, player)
		return status


	def get_valid_moves(self):
		valid_cells = []
		for r in range(self.size):
			for c in range(self.size):
				if self.board[r][c] == 0:
					valid_cells.append((r, c))
		return valid_cells


	def get_state(self):
		return self.board