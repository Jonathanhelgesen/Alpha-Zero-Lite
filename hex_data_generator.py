from Games.Hex.HexState import HexState
from MCTS.MCTS import MCTS
from MCTS.Node import Node
from NeuralNetworks.ANET import ANET
from NeuralNetworks.CNN import CNN
from NeuralNetworks.ResNet import ResNet
import random
import numpy as np
from datetime import datetime
import os
dirname = os.path.dirname(__file__)

def one_hot_tuple(lst):
    new_lst = []
    for i in range(len(lst)):
        row = []
        for j in range(len(lst[0])):
            if lst[i][j] == 0:
                row.append([0, 0])
            elif lst[i][j] == 1:
                row.append([1, 0])
            elif lst[i][j] == 2:
                row.append([0, 1])
        new_lst.append(row)
    return new_lst

def one_hot_pids(lst):
    new_lst = []
    for i in range(len(lst)):
        if lst[i] == 1:
            new_lst.append(np.array([1, 0]))
        elif lst[i] == 2:
            new_lst.append(np.array([0, 1]))
    return new_lst

def one_hot_boards(lst):
    """Takes a list of board states and replaces 0s, 1s and 2s with
    [0, 0], [1, 0] and [0, 1]"""
    new_lst = []
    for board in lst:
        new_board = []
        for row in board:
            new_row = []
            for i in range(len(row)):
                if row[i] == 0:
                    new_row.append([0, 0])
                elif row[i] == 1:
                    new_row.append([1, 0])
                elif row[i] == 2:
                    new_row.append([0, 1])
            new_board.append(new_row)
        new_lst.append(new_board)
    return new_lst


layers = [1024, 1024, 1024]
anet = ResNet(3)


board = [[0, 0, 0, 0, 0], 
 		 [0, 0, 0, 0, 0], 
 		 [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0]]

board = [[0, 0, 0], 
 		 [0, 0, 0], 
         [0, 0, 0]]


params = {
	'num_simulations': 500,
	'C': 1.4
}
mcts = MCTS(params)
results = []

starting_player = 1

board_data = []
player_data = []
distribution_data = []

tuple_data = []

for i in range(100):
    
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
        
tuple_training = np.array(one_hot_tuple(tuple_data))
dist_training = np.array(distribution_data)
board_training = np.array(one_hot_boards(board_data))
pid_training = np.array(one_hot_pids(player_data))

layers = [1024, 1024, 1024]
#anet = ANET(5, layers, 'softmax', 'adam', 'categorical_crossentropy')

anet.fit([board_training, pid_training], dist_training)
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
"""

input_1 = np.array([[[[0, 0], [0, 1], [0, 0]],
                    [[0, 0], [1, 0], [1, 0]],
                    [[0, 0], [0, 0], [0, 1]]]])

input_2 = np.array([[1, 0]])

# Get the model's prediction for the single sample
prediction = anet.predict(input_1, input_2)

# Print the prediction
print(prediction)