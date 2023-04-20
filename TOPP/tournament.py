from keras.models import load_model
from RL.Actor import Actor
from NeuralNetworks.ANET import ANET
from NeuralNetworks.CNN import CNN
from NeuralNetworks.CNN_plain import CNN_plain
from NeuralNetworks.cnn_encoded import CNN_encoded
from Games.Hex.HexState import HexState
from Games.Hex.HexVisualizer import visualize
import random
import datetime
from params import params
import os


class Tournament:
    

	def __init__(self, M, G, board_size):
		self.M = M #Number of saved ANETs
		self.G = G #Number of games between each actor
		self.board_size = board_size
		self.result_matrix = [[0 for j in range(M)] for i in range(M)]

	def get_actors(self, saved_nets):
		actors = []
		for net in saved_nets:
			model = load_model(net)
			#if net[0] == 'a':
			#anet = ANET(model)
			#elif net[0] == 'c':
			#anet = CNN(model)
			anet = CNN_plain(model)
			actors.append(Actor(anet))
		return actors


	def run(self, saved_nets):
		actors = self.get_actors(saved_nets)
		num_actors = len(actors)
		wins = [0] * len(actors)
		current_time = datetime.datetime.now().strftime("%m%d%H%M%S")
		visuals_path = f'TOPP_visuals\{current_time}_visuals'
		os.mkdir(visuals_path)
		for i in range(num_actors):
			for j in range(i + 1, num_actors):
				actor1 = actors[i]
				actor2 = actors[j]
				actor1_starts = True
				for g in range(self.G):
					
					actor1_type = None
					actor2_type = None
					if actor1_starts:
						player1 = actor1
						actor1_type = 1
						player2 = actor2
						actor2_type = 2
						actor1_starts = False
					else:
						player1 = actor2
						actor2_type = 1
						player2 = actor1
						actor1_type = 2
						actor1_starts = True
					
					actor1_name = saved_nets[i].split('\\')[-1]
					actor2_name = saved_nets[j].split('\\')[-1]
					print(f'{actor1_name} is player {actor1_type}')
					print(f'{actor2_name} is player {actor2_type}')
					board = HexState.get_empty_board(self.board_size)
					state = HexState(board, 1)
					turn = 0
					winner = state.get_winner()
					while winner == 0:
						action = None
						if turn % 2 == 0:
							#action = player1.get_top_2_action(state)
							action = player1.get_stochastic_action(state)
						else:
							#action = player2.get_top_2_action(state)
							action = player2.get_stochastic_action(state)
						state = state.get_child_state(action)

						if params['verbose']:
							visualize(state.get_board(), visuals_path, f'game_{i}_vs_{j}_{g}', True, params['show_time'])

						winner = state.get_winner()
						turn += 1
					if winner == actor1_type:
						wins[i] += 1
						self.result_matrix[i][j] += 1
					else:
						wins[j] += 1
						self.result_matrix[j][i] += 1

					print(f'Current standings: {wins}')
		win_percentages = []
		win_probs = []
		total = sum(wins)
		for player in wins:
			win_percentages.append(player/total)
			win_probs.append(player/(self.G*(self.M-1)))
		print(f'Percentage of total wins: {win_percentages}')
		print(f'Win percentages: {win_probs}')
		print('Win matrix:')
		for row in self.result_matrix:
			print(row)
					


