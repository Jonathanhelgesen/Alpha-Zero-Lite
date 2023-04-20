from NeuralNetworks.ANET import ANET
from NeuralNetworks.CNN import CNN
from NeuralNetworks.CNN_plain import CNN_plain
from NeuralNetworks.cnn_encoded import CNN_encoded
from NeuralNetworks.RBUF import RBUF
from NeuralNetworks.RBUF2 import RBUF2
from MCTS.OPMCTS import OPMCTS
from MCTS.Node import Node
from Games.Hex.HexState import HexState
from RL.Actor import Actor
import math
import time
import numpy as np
from keras.models import load_model
from Games.Hex.HexVisualizer import visualize
from params import params
import datetime
import os


class Simulator:
	"""Class that produces ANETs"""
    
	def __init__(self):
		self.RBUF = RBUF2()
		self.save_anet = True
		self.cases_added = 0

	
	def run(self, save_interval, num_anets, batch_size):
		# Etter hvert game, add cases til RBUF og . Når antall cases added er større eller lik save interval, lagre modellen
		# ANET default policy fit-es etter hvert game
		layers = [32]
		anet = CNN_plain()
		anet.make_model(params['board_size'], layers, 'relu', 'categorical_crossentropy')
		#anet = load_model('anet_4')
		actor = Actor(anet)

		current_time = datetime.datetime.now().strftime("%m%d%H%M%S")

		size = params['board_size']


		nets_path = f'simulations\{current_time}_nets_{size}'
		os.mkdir(nets_path)

		if params['verbose']:
			visuals_path = f'simulations\{current_time}_visuals_{size}'
			os.mkdir(visuals_path)


		# Save initial ANET
		actor.ANET.save(f'{nets_path}\init_cnn.h5')

		save_count = 0
		finished = False

		board = HexState.get_empty_board(params['board_size'])

		"""
		params = {
			'num_simulations': 5000,
			'C': 1.4,
    		'epsilon': 0.3
		}
		"""
                
		mcts = OPMCTS(actor, 1)
		game_count = 0
		starting_player = 1

		results = []

		#for _ in range(params['num_episodes']):
		while True:
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

				if params['verbose']:
					visualize(node.state.get_board(), visuals_path, f'Game{game_count}_turn{turn}', True, params['show_time'])

				#self.RBUF.add_case(state, distribution)
				self.RBUF.add_case(node.parent.state.get_board(), node.parent.state.current_player, distribution)
				self.cases_added += 1
				#visualize(node.state.get_board(), 'visuals', f'{game_count}{turn}')

			#print(f'Starting player: {starting_player}, Winning player: {node.state.get_winner()}')
			#print(f'{node.state.get_board()[0]}\n{node.state.get_board()[1]}\n{node.state.get_board()[2]}')

			states, targets = self.RBUF.get_state_and_target_batch(params['batch_size'])

			#if game_count % params['num_episodes'] == 0:
			actor.ANET.fit(states, targets)

			if (game_count) % params['save_interval'] == 0 and game_count != 1:
				actor.ANET.save(f'{nets_path}\\anet{save_count + 1}.h5')
				save_count += 1

			if save_count+1 == params['num_anets']:
				break


		# TODO: Lagre ANETs for TOPP