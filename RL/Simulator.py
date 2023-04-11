from NeuralNetworks.ANET import ANET
from NeuralNetworks.RBUF import RBUF
from MCTS.OPMCTS import OPMCTS
from MCTS.Node import Node
from Games.Hex.HexState import HexState
from RL.Actor import Actor
import numpy as np


class Simulator:
	"""Class that produces ANETs"""
    
	def __init__(self):
		self.RBUF = RBUF()
		self.save_anet = True
		self.cases_added = 0

	
	def run(self, save_interval, num_anets, batch_size):
		# Etter hvert game, add cases til RBUF og . Når antall cases added er større eller lik save interval, lagre modellen
		# ANET default policy fit-es etter hvert game
		layers = [16, 16]
		anet = ANET(3, layers, 'relu', 'categorical_crossentropy')
		actor = Actor(anet)
		save_count = 0
		finished = False

		board = HexState.get_empty_board(3)

		params = {
			'num_simulations': 50,
			'C': 1.4,
    		'epsilon': 0.1
		}
                
		mcts = OPMCTS(params, actor)
		game_count = 0

		starting_player = 1

		board_data = []
		player_data = []
		distribution_data = []

		tuple_data = []

		while True:
    
			if starting_player == 1:
				s0 = HexState(board, 1)
				starting_player = 2
			else:
				s0 = HexState(board, 2)
				starting_player = 1
    
			node = Node(s0)
			turn = 0


			while node.state.get_winner() == 0:
				print(f'Game {game_count}, turn {turn}')
				#root = node
				turn += 1
				if node.state.current_player == 1:
					node = mcts.select_action(node, node.state.current_player)
				elif node.state.current_player == 2:
					node = mcts.select_action(node, node.state.current_player)
				print(node.state.get_winner())
				#data.append([node.parent.state.board, node.parent.get_list_distribution(), node.parent.state.current_player])
				data_tuple = sum(node.parent.state.board, [])
				data_tuple.insert(0, node.parent.state.current_player)
				tuple_data.append(data_tuple)
				#board_data.append(sum(node.parent.state.board, []))
				board_data.append(node.parent.state.board)
				player_data.append(node.parent.state.current_player)
				distribution_data.append(sum(node.parent.get_list_distribution(), []))

				self.RBUF.add_case([tuple_data, distribution_data])
				self.cases_added += 1
				if self.cases_added % save_interval == 0:
					actor.ANET.save(f'anet_{save_count}')
					save_count += 1

			states, targets = self.RBUF.get_state_and_target_batch(batch_size)
			print(f'States: {states}')
			print(f'Targets: {targets}')
			actor.ANET.fit(states, targets)

			if save_count == num_anets:
				break


		# TODO: Lagre ANETs for TOPP