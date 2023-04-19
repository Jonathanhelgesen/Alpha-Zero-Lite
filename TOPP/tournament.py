from keras.models import load_model
from RL.Actor import Actor
from NeuralNetworks.ANET import ANET
from NeuralNetworks.CNN import CNN
from NeuralNetworks.CNN_plain import CNN_plain
from NeuralNetworks.cnn_encoded import CNN_encoded
from Games.Hex.HexState import HexState
from Games.Hex.HexVisualizer import visualize
import random


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
	
	
	def test_single_move(self, actor):
		model = load_model(actor)
		board = [[0, 0, 0, 0],
	   			 [0, 0, 0, 0],
				 [0, 0, 0, 0],
				 [0, 0, 0, 0]]
		state = HexState(board, 1)

	
	def worst_vs_best(self, saved_nets):
		actors = self.get_actors(saved_nets)
		wins = [0, 0]
		actor1 = actors[2]
		actor2 = actors[-3]
		actor1_starts = True
		for _ in range(10):
			player1 = None
			player2 = None
			if actor1_starts:
				player1 = actor1
				player2 = actor2
				actor1_starts = False
			else:
				player1 = actor2
				player2 = actor1
				actor1_starts = True
			board = HexState.get_empty_board(self.board_size)
			state = HexState(board, 1)
			turn = 0
			winner = None
			while state.get_winner() == 0:
				action = None
				if turn <= 2:
					action = random.choice(state.get_valid_moves())
				if turn % 2 == 0:
					action = player1.get_greedy_action(state)
				else:
					action = player2.get_greedy_action(state)
				state = state.get_child_state(action)
				winner = state.get_winner()
				turn += 1
			print(winner)
			if (winner == 1 and not actor1_starts) or winner == 2 and actor1_starts:
				wins[0] += 1
			else:
				wins[1] += 1
		print(wins)


	def test_predictions(self, a1):
		#actors = self.get_actors(nets)
		#actor1 = actors[0]
		anet = CNN()
		anet.make_model(3, [64], 'relu', 'categorical_crossentropy')
		#anet = load_model('anet_4')
		actor2 = Actor(anet)
		actor1 = a1
		board = HexState.get_empty_board(self.board_size)
		state = HexState(board, 1)
		turn = 0
		winner = state.get_winner()
		while winner == 0:
			action = None
			if turn % 2 == 0:
				predictions = actor1.get_predictions(state)
				action = actor1.get_stochastic_action(state)
			else:
				predictions = actor2.get_predictions(state)
				action = actor2.get_stochastic_action(state)
			print(f'{predictions[0]},\t{predictions[1]},\t{predictions[2]}')
			print(f'{predictions[3]},\t{predictions[4]},\t{predictions[5]}')
			print(f'{predictions[6]},\t{predictions[7]},\t{predictions[8]}')
			print(f'Action chosen: {action}\n')
			state = state.get_child_state(action)
			visualize(state.get_board(), 'visuals', f'3x3_{turn}')
			winner = state.get_winner()
			turn += 1

	def test_trained_net(self, a1):
		#actors = self.get_actors(nets)
		#actor1 = actors[0]
		anet = CNN()
		anet.make_model(3, [64], 'relu', 'categorical_crossentropy')
		#anet = load_model('anet_4')
		actor1 = Actor(anet)
		actor2 = a1
		wins = [0, 0]
		actor1_starts = True
		for i in range(self.G):
			if actor1_starts:
				player1 = actor1
				player2 = actor2
				actor1_type = 1
				actor1_starts = False
			else:
				player1 = actor2
				player2 = actor1
				actor1_type = 2
				actor1_starts = True
			board = HexState.get_empty_board(self.board_size)
			state = HexState(board, 1)
			turn = 0
			winner = state.get_winner()
			counter = 0
			while winner == 0:
				counter += 1
				action = None
				if turn % 2 == 0:
					predictions = actor2.get_predictions(state)
					if actor1_type == 2 and counter == 1:
						print(f'predictions: {predictions}')
					action = player1.get_stochastic_action(state)
				else:
					predictions = actor2.get_predictions(state)
					action = player2.get_stochastic_action(state)
				state = state.get_child_state(action)
				winner = state.get_winner()
				turn += 1
			
			if winner == actor1_type:
				wins[0] += 1
			else:
				wins[1] += 1
		print(f'Wins: {wins}')



	def run(self, saved_nets):
		actors = self.get_actors(saved_nets)
		num_actors = len(actors)
		wins = [0] * len(actors)
		for i in range(num_actors):
			for j in range(i + 1, num_actors):
				actor1 = actors[i]
				actor2 = actors[j]
				actor1_starts = True
				for _ in range(self.G):
					
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
					
					#player1 = actor1
					#player2 = actor2
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
						winner = state.get_winner()
						turn += 1
					print(f'i: {i}')
					print(f'j: {j}')
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
			win_probs.append(player/self.G)
		print(win_percentages)
		print(win_probs)
		for row in self.result_matrix:
			print(row)
					


