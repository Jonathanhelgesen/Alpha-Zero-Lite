from NeuralNetworks.ANET import ANET
from NeuralNetworks.CNN import CNN
from NeuralNetworks.RBUF import RBUF
from NeuralNetworks.RBUF2 import RBUF2
from MCTS.OPMCTS import OPMCTS
from MCTS.Node import Node
from Games.Hex.HexState import HexState
from RL.Actor import Actor
import math
import numpy as np
from keras.models import load_model
import os
dirname = os.path.dirname(__file__)


class Generator:
	"""Class that produces ANETs"""
    
	def __init__(self):
		self.RBUF = RBUF2()
		self.save_anet = True
		self.cases_added = 0


	def write_to_file(self, board, player, dist):
		file_path = os.path.join(os.getcwd(), 'my_data4.txt')

		# write the data to the text file
		file = open(file_path, mode='a')
		# write each case to the file
		line = str(board) + '+' + str(player) + '+' + str(dist) + '\n'
		file.write(line)

		# close the file
		file.close()

	
	def run(self):
		# Etter hvert game, add cases til RBUF og . Når antall cases added er større eller lik save interval, lagre modellen
		# ANET default policy fit-es etter hvert game
		layers = [64]
		anet = CNN()
		anet.make_model(4, layers, 'relu', 'categorical_crossentropy')
		#anet = load_model('anet_4')
		actor = Actor(anet)

		board = HexState.get_empty_board(4)

		params = {
			'num_simulations': 6401,
			'C': 1.4,
    		'epsilon': 1
		}
                
		mcts = OPMCTS(params, actor, 10)
		game_count = 0
		starting_player = 1

		results = []

		while True:
		#for _ in range(5):
			game_count += 1
			if starting_player == 1:
				s0 = HexState(board, 1)
				starting_player = 2
			else:
				s0 = HexState(board, 2)
				starting_player = 1
    
			node = Node(s0)
			turn = 0

			# List of states and distributions for training ANET after each game
			game_states = []
			game_distributions = []

			while node.state.get_winner() == 0:
				print(f'Game {game_count}, turn {turn}')
				turn += 1
				if node.state.current_player == 1:
					node = mcts.select_action(node, node.state.current_player)
				elif node.state.current_player == 2:
					node = mcts.select_action(node, node.state.current_player)
				print(node.state.get_winner())

				state = sum(node.parent.state.board, [])
				state.insert(0, node.parent.state.current_player)
				distribution = sum(node.parent.get_list_distribution(), [])

				game_states.append(state)
				game_distributions.append(distribution)

				#self.RBUF.add_case(state, distribution)
				self.write_to_file(node.parent.state.get_board(), node.parent.state.current_player, distribution)
		
			results.append(starting_player + node.state.get_winner())
			print(results)