import ast
from NeuralNetworks.CNN import CNN
from NeuralNetworks.CNN2 import CNN2
from NeuralNetworks.CNN3 import CNN3
from NeuralNetworks.CNN4 import CNN4
from NeuralNetworks.ANET import ANET
from NeuralNetworks.CNN_plain import CNN_plain
from NeuralNetworks.cnn_encoded import CNN_encoded
from RL.Actor import Actor
from NeuralNetworks.RBUF2 import RBUF2
from TOPP.tournament import Tournament
from Games.Hex.HexState import HexState
import random
import time

random.seed(43)

data_list = []
RBUF = RBUF2()
RBUF_test = RBUF2()

with open('my_data3.txt', 'r') as f:
    for line in f:
        elements = line.strip().split('+')
        two_d_list = ast.literal_eval(elements[0])
        integer = int(elements[1])
        float_list = [float(i) for i in ast.literal_eval(elements[2])]
        #if random.random() < 1:
        RBUF.add_case(two_d_list, integer, float_list)
        #else:
            #RBUF_train.add_case(two_d_list, integer, float_list)

        
layers = [64]
anet = CNN_plain(layers)
anet.make_model(3, layers, 'relu', 'categorical_crossentropy')
#anet = load_model('anet_4')
actor = Actor(anet)

states, targets = RBUF.get_state_and_target_batch(25000)
actor.ANET.fit(states, targets)

"""
board = [[0, 0, 0, 0],
 		 [0, 0, 2, 0],
 		 [0, 1, 1, 0],
 		 [0, 0, 2, 0]]
"""

board = [[0, 0, 0],
         [0, 0, 0],
         [0, 0, 0]]

player = 2

predictions = actor.ANET.predict(board, player)
print(predictions)

board = [[0, 0, 0],
         [0, 0, 0],
         [0, 0, 0]]

player = 1

predictions = actor.ANET.predict(board, player)
print(predictions)

#states, targets = RBUF_test.get_state_and_target_batch(25000)
#actor.ANET.test(states, targets)

#actor.ANET.save(f'initial_3x3')
#best_action = actor.get_greedy_action([board, player])


t = Tournament(2, 1000, 3)
#t.test_trained_net(actor)


# Save initial ANET
actor.ANET.save('ann_3x3_0.h5')


time.sleep(5)


anet2 = CNN_plain()
anet2.make_model(3, [64], 'relu', 'categorical_crossentropy')
#anet = load_model('anet_4')
actor1 = Actor(anet2)
wins = [0, 0]
actor1_starts = True
for i in range(100):
	if actor1_starts:
		player1 = actor1
		player2 = actor
		actor1_type = 1
		actor1_starts = False
	else:
		player1 = actor
		player2 = actor1
		actor1_type = 2
		actor1_starts = True
	board = HexState.get_empty_board(3)
	state = HexState(board, 1)
	turn = 0
	winner = state.get_winner()
	counter = 0
	while winner == 0:
		counter += 1
		action = None
		if turn % 2 == 0:
			predictions = actor.get_predictions(state)
			predictions = player1.ANET.predict(state.get_board(), state.current_player)
			if actor1_type == 2 and counter == 1:
				print(actor.ANET.name)
				print(f'predictions: {predictions}')
			action = player1.get_stochastic_action(state)
		else:
			predictions = actor.get_predictions(state)
			action = player2.get_stochastic_action(state)
		state = state.get_child_state(action)
		winner = state.get_winner()
		turn += 1
	
	if winner == actor1_type:
		wins[0] += 1
	else:
		wins[1] += 1
print(f'Wins: {wins}')