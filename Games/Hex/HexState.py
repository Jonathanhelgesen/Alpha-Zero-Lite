from collections import deque
import copy
import numpy as np
import math
import random

class HexState():

	def __init__(self, board, player, move=None, turn=0, size=None): #fjernet move, kan legges til
		if size is not None:
			self.size = size
			self.board = [[0]*size]*size
		else:
			self.size = len(board)
			self.board = board
		self.current_player = player
		self.turn = turn
		self.move = copy.copy(move)

	def check_full_board(self):
		"""
    	Checks if a 2D list is filled with no 0s and that the number of 1s and -1s are equal.
    	"""
		for row in self.board:
			for element in row:
				if element == 0:
					return False
		return True


	def get_valid_moves(self):
		valid_cells = []
		for r in range(len(self.board)):
			for c in range(len(self.board)):
				if self.board[r][c] == 0:
					valid_cells.append((r, c))
		return valid_cells
	

	def execute_move(self, move):
		self.move = copy.copy(move)
		r = move[0]
		c = move[1]
		if self.board[r][c] != 0:
			raise Exception('ILLEGAL MOVE!')
		self.board[r][c] = self.next_player()

	
	def next_player(self):
		if self.current_player == 1:
			return 2
		elif self.current_player == 2:
			return 1


	def generate_child_states(self):
		child_states = []
		valid_moves = self.get_valid_moves()
		for move in valid_moves:
			board_copy = copy.deepcopy(self.board)
			child_state = HexState(board_copy, self.next_player(), move=move, turn=self.turn+1)
			child_state.execute_move(move)
			child_states.append(child_state)
		return child_states
	

	def get_flat_state(self):
		flat_state = []
		flat_state.append(self.current_player)
		flat_board = sum(self.board, [])
		flat_state.extend(flat_board)
		return flat_state
		

	def get_winner(self):
		if self.turn < 2 * self.size - 1:
			return 0
		matrix = copy.deepcopy(self.board)
		queue = deque()
		visited = set()
		max_turns = self.size**2
		# Check if there is a path from right to left
		if self.current_player == 2 or self.turn == max_turns:
			for r in range(self.size):
				if matrix[r][0] == 1:
					queue.append((r,0))
					visited.add((r,0))
			while queue:
				r, c = queue.popleft()
				if c == len(matrix) - 1:
					return 1
				for x, y in ((r-1,c), (r,c-1), (r+1,c), (r,c+1), (r-1,c+1), (r+1,c-1)):
					if 0 <= x < len(matrix) and 0 <= y < len(matrix[0]) and matrix[x][y] == 1 and (x,y) not in visited:
						queue.append((x,y))
						visited.add((x,y))
		if self.current_player == 1 or self.turn == max_turns:
			# Check if there is a path from top to bottom
			for c in range(self.size):
				if matrix[0][c] == 2:
					queue.append((0,c))
					visited.add((0,c))
			while queue:
				r, c = queue.popleft()
				if r == len(matrix) - 1:
					return 2
				for x, y in ((r-1,c), (r,c-1), (r+1,c), (r,c+1), (r-1,c+1), (r+1,c-1)):
					if 0 <= x < len(matrix) and 0 <= y < len(matrix[0]) and matrix[x][y] == 2 and (x,y) not in visited:
						queue.append((x,y))
						visited.add((x,y))
		# Return 0 if no winner yet
		return 0


if __name__ == "__main__":
	board = [[0, 0, 0, 0, 2],
		 [2, 2, 2, 2, 2],
		 [1, 1, 1, 2, 1],
		 [0, 0, 0, 2, 0],
		 [0, 0, 0, 2, 0]]
         
	state = HexState(board, 1, turn=14)
	print(state.get_winner())