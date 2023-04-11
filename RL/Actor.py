import random

class Actor:
	"""Class for handling input states and output actions"""
    
	def __init__(self, ANET):
		self.ANET = ANET

	def set_anet(self, ANET):
		self.ANET = ANET

	def get_greedy_action(self, state):
		"""Returns action with the highest score from the ANET"""
		flat_state = state.get_flat_state()
		flat_board = flat_state[1:]
		predictions = self.ANET.predict(flat_state).tolist()[0]
		max_prob = 0
		action_index = None
		for i in range(len(predictions)):
			if predictions[i] > max_prob and flat_board[i] == 0:
				max_prob = predictions[i]
				action_index = i
		action = state.flat_index_to_move(action_index)
        # New node based on ANET prediction
		return action
	
	def get_epsilon_greedy_action(self, state, epsilon):
		flat_state = state.get_flat_state()
		flat_board = flat_state[1:]
		if random.random() < epsilon:
			indices = [i for i in range(len(flat_board)) if flat_board[i] == 0]
			action_index = random.choice(indices)
		else:
			predictions = self.ANET.predict(flat_state).tolist()[0]
			max_prob = 0
			action_index = None
			for i in range(len(predictions)):
				if predictions[i] > max_prob and flat_board[i] == 0:
					max_prob = predictions[i]
					action_index = i
		action = state.flat_index_to_move(action_index)
		return action