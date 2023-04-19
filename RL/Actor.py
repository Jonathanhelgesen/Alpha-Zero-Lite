import random
from NeuralNetworks.ANET import ANET
from NeuralNetworks.CNN import CNN
from NeuralNetworks.CNN_plain import CNN_plain
from NeuralNetworks.cnn_encoded import CNN_encoded
import numpy as np

class Actor:
	"""Class for handling input states and output actions"""
    
	def __init__(self, ANET):
		self.ANET = ANET

	def set_anet(self, ANET):
		self.ANET = ANET

	def get_predictions(self, state):
		board = state.get_board()
		player = state.current_player
		return self.ANET.predict(board, player).tolist()[0]


	def get_greedy_action(self, state):
		"""Returns action with the highest score from the ANET"""
		#flat_state = state.get_flat_state()
		#flat_board = flat_state[1:]
		#predictions = self.ANET.predict(flat_state).tolist()[0]
		board = state.get_board()
		player = state.current_player
		predictions = self.ANET.predict(board, player).tolist()[0]
		flat_board = state.get_flat_state().pop(0)
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
		predictions = []
		if random.random() < epsilon:
			indices = [i for i in range(len(flat_board)) if flat_board[i] == 0]
			action_index = random.choice(indices)
		else:
			#predictions = self.ANET.predict(flat_state).tolist()[0]
			board = state.get_board()
			player = state.current_player
			predictions = self.ANET.predict(board, player).tolist()[0]
			max_prob = 0
			action_index = None
			for i in range(len(predictions)):
				if predictions[i] > max_prob and flat_board[i] == 0:
					max_prob = predictions[i]
					action_index = i
		try:
			action = state.flat_index_to_move(action_index)
		except:
			print(f'Predictions: {predictions}')
			print(f'Board: {state.get_board()}')
		return action
	
	def get_top_2_action(self, state):
		flat_state = state.get_flat_state()
		flat_board = flat_state[1:]
		#predictions = self.ANET.predict(flat_state).tolist()[0]
		board = state.get_board()
		player = state.current_player
		predictions = None
		if isinstance(self.ANET, ANET):
			predictions = self.ANET.predict(flat_state).tolist()[0]
		elif isinstance(self.ANET, CNN):
			predictions = self.ANET.predict(board, player).tolist()[0]#numpy().tolist()
		max_prob_1 = 0
		max_prob_2 = 0
		action_index_1 = None
		action_index_2 = None
		for i in range(len(predictions)):
			if predictions[i] > max_prob_1 and flat_board[i] == 0:
				max_prob_1 = predictions[i]
				action_index_1 = i
			elif predictions[i] > max_prob_2 and flat_board[i] == 0:
				max_prob_2 = predictions[i]
				action_index_2 = i
		choose_best_action_prob = max_prob_1 / (max_prob_1 + max_prob_2)**2
		if choose_best_action_prob < random.random() or action_index_2 is None:
			return state.flat_index_to_move(action_index_1)
		else:
			return state.flat_index_to_move(action_index_2)


	def get_stochastic_action(self, state):
		flat_state = state.get_flat_state()
		flat_board = flat_state[1:]

		# Step 1: Create a list of indices where integers is not equal to 0
		empty_indexes = [i for i in range(len(flat_board)) if flat_board[i] == 0]

		board = state.get_board()
		player = state.current_player

		predictions = None
		if isinstance(self.ANET, ANET):
			predictions = self.ANET.predict(flat_state).tolist()[0]
		elif isinstance(self.ANET, CNN):
			predictions = self.ANET.predict(board, player).tolist()[0]
		elif isinstance(self.ANET, CNN_plain):
			predictions = self.ANET.predict(board, player).tolist()[0]

		# Step 2: Extract the probabilities corresponding to the nonzero indices
		action_probabilities = [predictions[i]**2 for i in empty_indexes]

		# Step 3: Normalize the probabilities so that they sum to 1
		action_probabilities_normalized = np.array(action_probabilities) / sum(action_probabilities)

		# Step 4: Sample an index based on the probabilities
		index = random.choices(empty_indexes, weights=action_probabilities_normalized)[0]
		return state.flat_index_to_move(index)