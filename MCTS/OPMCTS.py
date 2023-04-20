import math
import random
import time
from params import params
#from Node import Node

class OPMCTS:
    
    def __init__(self, actor, time_limit=10):
        self.actor = actor
        self.time_limit = params['time_limit']


    def select_action(self, node, starting_player, is_random=False):
        # Continue grow the given node
        self.generate_tree(node, params['num_rollouts'], starting_player)

        if is_random:
            return random.choice(node.children)

        current_player = node.state.current_player
        best_child = None

		# Find node with best expected final result
        highest_q = -float('inf')
        lowest_q = float('inf')

        for child in node.children:
            # Check ratio between visits and wins
            q_value = child.wins/(child.visits + 1)  

			# Best move depends on which player is moving
            if starting_player == current_player:
                if q_value > highest_q:
                    best_child = child
                    highest_q = q_value
            else: 
                if q_value < lowest_q:
                    best_child = child
                    lowest_q = q_value
                    
        return best_child


    def generate_tree(self, node, num_simulations, current_player):
        start_time = time.time()
        for i in range(num_simulations):
            
			# Find best child node using tree policy
            best_node = self.tree_search(node, current_player)
            
			# Generate child nodes
            best_node.expand()
            
			# If node isn't leaf node, get random child
            if len(best_node.get_children()) > 0: 
                best_node = random.choice(best_node.children)
                
			# Perform rollout
            winner = self.evaluate(best_node)
            self.backpropagate(best_node, winner, current_player)
            
            if time.time() - start_time > params['time_limit']:
                break
        print(f'Rollouts completed: {i}')

    def tree_search(self, node, current_player):
        if not node.children:  
            # Leaf node, end search
            return node

        best_child = node
        highest_value = float('-inf')
        lowest_value = float('inf')
        is_opponent = node.state.current_player != current_player

        for child in node.children:
            # Use tree policy to asses the value of the child
            value = self.tree_policy_value(node, child, is_opponent)
            if is_opponent and value < lowest_value:
                # Minimize for opponent
                best_child = child
                lowest_value = value

            elif (not is_opponent) and value > highest_value:
                # Maximize for current player
                best_child = child
                highest_value = value
        
		# Run until leaf node is reached
        return self.tree_search(best_child, current_player)
    

    def tree_policy_value(self, parent, child, is_opponent):
        q_value = child.wins / (child.visits + 1)
        # Upper Confidence Bound for exploration bonus
        u_value = math.sqrt(math.log(parent.visits) * params['C'] / (child.visits + 1))

		#Min-max
        if is_opponent:
            return q_value - u_value
        return q_value + u_value


    def evaluate(self, node):
        winner = node.state.get_winner()
        #random_simulation = random.random() < self.params['epsilon']
        while winner == 0:
            action = self.actor.get_epsilon_greedy_action(node.state, params['epsilon'])
            # New node based on ANET/random prediction
            node = node.get_child_node(action)
                
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