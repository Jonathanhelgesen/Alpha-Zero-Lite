from keras.models import load_model
from RL.Actor import Actor
from Games.Hex.HexState import HexState

class TOPP:
    

	def __init__(self, M, G, num_episodes, board_size):
		self.M = M #Number of saved ANETs
		self.G = G #Number of games between each actor
		self.num_episodes = num_episodes
		self.board_size = board_size

	def run_tournament(self, anets):
		actors = []
		for anet in anets:
			model = load_model(anet)
			actors.append(Actor(model))

	def get_opponent(self, player):
		if player == 1:
			return 2
		elif player == 2:
			return 1
	
	def fight(self, actor1, actor2, matches):
		actor1_starts = True
		actor1_is_1 = True
		actor2_is_1 = True
		for _ in matches:
			player = None
			starting_type = None
			if actor1_starts:
				player = 1
				actor1_starts = False
				if actor1_is_1:
					starting_type = 1
					actor1_is_1 = False
				else:
					starting_type = 2
					actor1_is_1 = True
			else:
				player = 2
				actor1_starts = True
				if actor2_is_1:
					starting_type = 1
					actor2_is_1 = False
				else:
					starting_type = 2
					actor1_is_2 = True
			


		def simulate_game(self, first_actor, first_actor_type, second_actor, second_actor_type):
			board = HexState.get_empty_board(self.board_size)
			state = HexState(board, player)

