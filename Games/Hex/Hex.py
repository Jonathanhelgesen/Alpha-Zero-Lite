from DisjointSet import DisjointSet
import sys
sys.path.append("..")
from Game import Game
from collections import deque


def check_win(matrix, player):
	queue = deque()
	visited = set()
	if player == 1:
		for r in range(len(matrix[0])):
			if matrix[r][0] == player:
				queue.append((r,0))
				visited.add((r,0))
		while queue:
			r, c = queue.popleft()
			if c == len(matrix) - 1:
				return player
			for x, y in ((r-1,c), (r,c-1), (r+1,c), (r,c+1), (r-1,c+1), (r+1,c-1)):
				if 0 <= x < len(matrix) and 0 <= y < len(matrix[0]) and matrix[x][y] == player and (x,y) not in visited:
					queue.append((x,y))
					visited.add((x,y))
		return 0
	else:
		for c in range(len(matrix[0])):
			if matrix[0][c] == player:
				queue.append((0,c))
				visited.add((0,c))
		while queue:
			r, c = queue.popleft()
			if r == len(matrix) - 1:
				return player
			for x, y in ((r-1,c), (r,c-1), (r+1,c), (r,c+1), (r-1,c+1), (r+1,c-1)):
				if 0 <= x < len(matrix) and 0 <= y < len(matrix[0]) and matrix[x][y] == player and (x,y) not in visited:
					queue.append((x,y))
					visited.add((x,y))
		return 0
	

def get_next_state(state, move, player=1):
	r = move[0]
	c = move[1]
	if 0 <= r < len(state) and 0 <= c < len(state):
		if state[r][c] == 0:
			state[r][c] = player
	return state

class Hex():

	def __init__(self, size):
        # TODO: Change 0, 1 and -1 to [0, 0], [1, 0] and [0, 1] for ANN
		self.size = size
		self.board = [[0] * size for _ in range(size)]
		self.cells = [(i, j) for i in range(size) for j in range(size)]


	def is_move_valid(self, move):
		r = move[0]
		c = move[1]
		if 0 <= r < self.size and 0 <= c < self.size:
			if self.board[r][c] == 0:
				return True
			else:
				return False
		else:
			return False


	def execute_move(self, move, player):
		"""Executes the given move, if allowed. Returns the status of the game"""
		r = move[0]
		c = move[1]

		if self.is_move_valid(move) and player in [1, -1]:
			self.board[r][c] = player
		else:
			raise ValueError('Move is not allowed')

		print(self.board)
		status = check_win(self.board, player)
		return status
	

	def check_move(self, state, move, player):
		r = move[0]
		c = move[1]

		if self.is_move_valid(move) and player in [1, -1]:
			state[r][c] = player
		else:
			raise ValueError('Move is not allowed')
		
		return check_win(state, player)


	def get_valid_moves(self):
		valid_cells = []
		for r in range(self.size):
			for c in range(self.size):
				if self.board[r][c] == 0:
					valid_cells.append((r, c))
		return valid_cells


	def get_state(self):
		return self.board


	def check_status(self, move, player):
		self.execute_move(move)
		status = self.check_win(self.board, player)
		self.undo_move(move)
		return status
	
	def get_move_count(self):
		return self.size**2