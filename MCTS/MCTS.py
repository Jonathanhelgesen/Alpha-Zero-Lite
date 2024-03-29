import math
import random
#from Node import Node

class MCTS:
    
    def __init__(self, params):
        self.params = params

    def select_action(self, node, starting_player, is_random=False):
        self.update(node, self.params['num_simulations'], starting_player)  # Update the tree using mcts

        if is_random:
            return random.choice(node.children)

        current_player = node.state.current_player
        action_node = None

        highest_qsa = -float('inf')
        lowest_qsa = float('inf')

        for child in node.children:
            qsa = float(child.wins)/float(child.visits)  # Calculate Q(s,a)

            if starting_player == current_player:
                if qsa > highest_qsa:
                    highest_qsa = qsa
                    action_node = child
            else:  # If the current player is the opposing player, the best score is the lowest Q(s,a)
                if qsa < lowest_qsa:
                    lowest_qsa = qsa
                    action_node = child
        return action_node

    def update(self, node, num_simulations, current_player):
        for _ in range(num_simulations):
            best_node = self.tree_search(node, current_player)
            best_node.expand()
            if len(best_node.get_children()) > 0:  # Choose a random child if just expanded
                best_node = random.choice(best_node.children)
            winner = self.evaluate(best_node)
            self.backpropagate(best_node, winner, current_player)

    def tree_policy_value(self, parent, child, is_opponent):
        q_value = child.wins / (child.visits + 1)
        u_value = self.params['C'] * math.sqrt(math.log(parent.visits) / (child.visits + 1))

        if is_opponent:
            return q_value - u_value
        return q_value + u_value

    def tree_search(self, node, current_player):
        if not node.children:  # Breaks recursion and returns the best leaf node
            return node

        best_child = node
        highest_value = float('-inf')
        lowest_value = float('inf')
        opposing_player = node.state.current_player != current_player

        for child in node.children:
            value = self.tree_policy_value(node, child, opposing_player)  # Get value of node based on the tree policy

            if opposing_player and value < lowest_value:
                # The best value is the lowest value when the player is the opposing player
                best_child = child
                lowest_value = value

            elif (not opposing_player) and value > highest_value:
                best_child = child
                highest_value = value
        return self.tree_search(best_child, current_player)  # Recursively search the tree until reaching best leaf node

    @staticmethod
    def evaluate(node):
        winner = node.state.get_winner()
        while winner == 0: #Kan optimaliseres
            node = node.get_random_child()
            winner = node.state.get_winner()
        return winner

    @staticmethod
    def backpropagate(node, winner, current_player):
        while node is not None:
            if winner == current_player:
                node.wins += 1
            node.visits += 1
            node = node.parent
            


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
        dist = [[0 for i in range(self.state.size)] for j in range(self.state.size)]
        visits_sum = sum(child.visits for child in self.children)
        for child in self.children:
            move = child.state.move
            dist[move[0]][move[1]] = child.visits / visits_sum
        return dist