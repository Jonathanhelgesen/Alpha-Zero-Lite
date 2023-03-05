from Hex import Hex
from HexVisualizer import *
import random
import os
from datetime import datetime
dirname = os.path.dirname(__file__)


if __name__ == "__main__":
    size = int(input('Size of Hex board: '))
    hex = Hex(size)
    turns = 0

    game_name = datetime.now().strftime('%d%m%Y_%H%M%S')
    os.mkdir(f'{dirname}\\visuals\\{game_name}')

    winner = 0

    while winner == 0:
        if turns % 2 == 0:
            print(hex.get_state())
            row = int(input('Select row: '))
            column = int(input('Select column: '))
            winner = hex.execute_move([row, column], 1)
            visualize(hex.get_state(), f'visuals\\{game_name}', f'move_{turns}')
        else:
            move = random.sample(hex.get_valid_moves(), 1)[0]
            winner = hex.execute_move(move, -1)
            visualize(hex.get_state(), f'visuals\\{game_name}', f'move_{turns}')
        turns += 1
    
    print(f'Winner is player {winner}')

#def save_visualization(board):
