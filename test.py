from MCTS.OPMCTS import OPMCTS
from MCTS.Node import Node
from NeuralNetworks.ANET import ANET
from Games.Hex.HexState import HexState
from RL.Actor import Actor
import numpy as np


layers = [16, 16]
anet = ANET(3, layers, 'relu', 'categorical_crossentropy')
actor = Actor(anet)


board = HexState.get_empty_board(3)

params = {
	'num_simulations': 100,
	'C': 1.4,
    'epsilon': 0.7
}
mcts = OPMCTS(params, actor)
results = []

starting_player = 1

board_data = []
player_data = []
distribution_data = []

tuple_data = []

for i in range(10):
    
    if starting_player == 1:
        s0 = HexState(board, 1)
        starting_player = 2
    else:
        s0 = HexState(board, 2)
        starting_player = 1
    
    node = Node(s0)
    turn = 0


    while node.state.get_winner() == 0:
        print(f'Game {i}, turn {turn}')
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
        
#tuple_training = np.array(one_hot_tuple(tuple_data))
#dist_training = np.array(distribution_data)
#board_training = np.array(one_hot_boards(board_data))
#pid_training = np.array(one_hot_pids(player_data))

layers = [1024, 1024, 1024]
#anet = ANET(5, layers, 'softmax', 'adam', 'categorical_crossentropy')

#anet.fit([board_training, pid_training], dist_training)
"""
input_data = np.array([[[1, 0], 
                  [0, 0], [0, 1], [0, 0], [0, 0], [0, 0], 
                  [0, 0], [1, 0], [1, 0], [1, 0], [1, 0], 
                  [0, 0], [0, 0], [0, 0], [0, 1], [0, 0], 
                  [0, 1], [0, 0], [0, 0], [0, 0], [0, 0],
                  [0, 0], [0, 0], [0, 1], [0, 0], [0, 0]]])


input_1 = np.array([[[[0, 0], [0, 1], [0, 0], [0, 0], [0, 0]],
                    [[0, 0], [1, 0], [1, 0], [1, 0], [1, 0]],
                    [[0, 0], [0, 0], [0, 0], [0, 1], [0, 0]],
                    [[0, 1], [0, 0], [0, 0], [0, 0], [0, 0]],
                    [[0, 0], [0, 0], [0, 1], [0, 0], [0, 0]]]])


input_1 = np.array([[[[0, 0], [0, 1], [0, 0]],
                    [[0, 0], [1, 0], [1, 0]],
                    [[0, 0], [0, 0], [0, 1]]]])

input_2 = np.array([[1, 0]])
"""

input_data = [1, 
              0, 2, 0, 
              1, 1, 0, 
              0, 0, 2]

# Get the model's prediction for the single sample
prediction = anet.predict(input_data)

# Print the prediction
print(prediction)