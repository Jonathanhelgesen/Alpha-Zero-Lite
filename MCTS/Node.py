import random
import copy
from Games.Hex.HexState import HexState

class Node:

    def __init__(self, state, parent=None):
        self.parent = parent
        self.state = state
        self.children = []
        self.wins = 0
        self.visits = 0

    def get_children(self):
        children = []
        for state in self.state.generate_child_states():
            child = Node(state, parent=self)
            children.append(child)
        return children

    def get_random_child(self):
        if self.children:
            return random.choice(self.children)
        else:
            return random.choice(self.get_children())
        
    def get_child_node(self, action):
        child_state = self.state.get_child_state(action)
        return Node(child_state, parent=self)
        
    def get_random_or_terminal_child(self):
        if not self.children:
            self.children = self.get_children()
        if self.state.turn < 2*self.state.size - 1:
            return random.choice(self.children)
        for child in self.children:
            if child.state.get_winner() == -child.state.current_player:
                return child
        return random.choice(self.children)

    def expand(self):
        if not self.children:
            self.children = self.get_children()
            
    def get_distribution(self):
        if not self.children:
            raise Exception('Node has no children')

        distribution = {}
        visits_sum = sum(child.visits for child in self.children)
        print(visits_sum)
        for child in self.children:
            #distribution[child.state.move] = child.visits / visits_sum
            distribution[child.state.move] = (child.visits, child.wins, visits_sum)
        return distribution
    
    def get_list_distribution(self):
        #next_move_wins, dist = self.check_next_move_win()
        #if next_move_wins:
            #return dist
        dist = [[0 for i in range(self.state.size)] for j in range(self.state.size)]
        visits_sum = sum(child.visits for child in self.children)
        for child in self.children:
            move = child.state.move
            dist[move[0]][move[1]] = child.visits / visits_sum
        return dist
    
    def check_next_move_win(self):
        next_move_wins = False
        valid_moves = self.state.get_valid_moves()
        dist = [[0 for i in range(self.state.size)] for j in range(self.state.size)]
        for move in valid_moves:
            child_state = self.state.get_child_state(move)
            if child_state.get_winner() == self.state.current_player:
                next_move_wins = True
                dist[move[0]][move[1]] = 0.9
            else:
                dist[move[0]][move[1]] = 0.01
        return next_move_wins, dist
    

    def check_next_move_win2(self):
        next_move_wins = False
        valid_moves = self.state.get_valid_moves()
        dist = [[0 for i in range(self.state.size)] for j in range(self.state.size)]
        for move in valid_moves:
            child_state = self.state.get_child_state(move)
            flat_move_index = move[0]*self.state.size + move[1]
        return next_move_wins, dist
            