import Node
import numpy as np

class MCTS:
    
	def __init__(self, params, game):
		self.params = params
		self.game = game

	def search(self, state):
		root = Node(self.game, self.args, state)

		for search in range(self.params['n_searches']):
			node = root

			while node.is_fully_expanded():
				node = node.select()

		value, is_terminal = self.game.get_value_and_terminated(node.state, node.action_taken) # Denne må implementeres, kan splittes i to
		value = self.game.get_opponent_value(value)	# Må også implementeres. Noe som hører til i en abstrakt klasse? Er logikk for alle spill

		if not is_terminal:
			node = node.expand()
			value = node.simulate()

		node.backpropagate(value)


		action_probabilities = np.zeros(self.game.action_size)
		for child in root.children:
			action_probabilities[child.action_taken] = child.visit_count

		action_probabilities /= np.sum(action_probabilities)
		return action_probabilities
