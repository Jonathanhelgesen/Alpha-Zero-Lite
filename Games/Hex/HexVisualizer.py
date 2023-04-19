import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
import math
import os
dirname = os.path.dirname(__file__)

matrix = [[0, 1, 0, 0],
		  [-1, 1, 0, 0],
		  [0, 0, 0, -1],
		  [1, 0, -1, 0]]

G = nx.Graph()

def add_nodes(G, matrix):
	for r in range(len(matrix)):
		for c in range(len(matrix[0])):
			color = 'gray'
			if matrix[r][c] == 1:
				color = 'red'
			elif matrix[r][c] == 2:
				color = 'blue'
			G.add_node(f'{r}_{c}', color=color)

def get_edges(matrix):
	edges = []
	for r in range(len(matrix)):
		for c in range(len(matrix[0])):
			# Flere forenklinger her?
			#if r - 1 >= 0:
				#edges.append((f'{r}_{c}', f'{r - 1}_{c}'))
			#if r - 1 >= 0 and c + 1 < len(matrix[0]):
				#edges.append((f'{r}_{c}', f'{r - 1}_{c + 1}'))
			if c - 1 >= 0:
				edges.append((f'{r}_{c}', f'{r}_{c - 1}'))
			#if c + 1 < len(matrix[0]):
				#edges.append((f'{r}_{c}', f'{r}_{c + 1}'))
			if r + 1 < len(matrix) and c - 1 >= 0:
				edges.append((f'{r}_{c}', f'{r + 1}_{c - 1}'))
			if r + 1 < len(matrix):
				edges.append((f'{r}_{c}', f'{r + 1}_{c}'))
	return edges

def get_positions(matrix):
	positions = {}
	rot_matrix = np.array([[math.cos(-3*math.pi/4), -math.sin(-3*math.pi/4)],
						   [math.sin(-3*math.pi/4), math.cos(-3*math.pi/4)]])
	for r in range(len(matrix)):
		for c in range(len(matrix[0])):
			product = np.matmul(rot_matrix, np.array([r, c]))
			positions.update({f'{r}_{c}': [product[0], product[1]]})

	return positions

def get_edge_colors(edges, matrix):
	n = len(matrix) - 1
	pairs = [[i, i+1] for i in range(n)]
	reversed_pairs = [[i, i-1] for i in range(n, 0, -1)]
	colors = []
	for edge in edges:
		node0_strings = edge[0].split('_')
		node0_ints = [eval(i) for i in node0_strings]
		node1_strings = edge[1].split('_')
		node1_ints = [eval(i) for i in node1_strings]

		row_pair = [node0_ints[0], node1_ints[0]]
		column_pair = [node0_ints[1], node1_ints[1]]

		if (column_pair in pairs or column_pair in reversed_pairs) and row_pair[0] == row_pair[1] and row_pair[0] in [0, n] and row_pair[1] in [0, n]:
			colors.append((edge[0], edge[1], {'edge_color': 'blue'}))
		elif (row_pair in pairs or row_pair in reversed_pairs) and column_pair[0] == column_pair[1] and column_pair[0] in [0, n] and column_pair[1] in [0, n]:
			colors.append((edge[0], edge[1], {'edge_color': 'red'}))
		else:
			colors.append((edge[0], edge[1], {'edge_color': 'black'}))

	return colors
	
		

def visualize(matrix, folder, name):
	G = nx.Graph()
	add_nodes(G, matrix)
	edges = get_edges(matrix)
	edges_with_color = get_edge_colors(edges, matrix)
	edges_without_color = [(x, y) for x, y, z in edges_with_color]
	colors = [z['edge_color'] for x, y, z in edges_with_color]
	G.add_edges_from(edges_without_color)
	node_colors = [node[1]['color'] for node in G.nodes(data=True)]
	ax = plt.figure().gca()
	ax.set_axis_off()
	nx.draw(G, get_positions(matrix), edge_color=colors, node_color=node_colors, with_labels=True, font_color='white', font_size=7)

	path = os.path.join(dirname, f'visuals\\{folder}')
	plt.savefig(f'{folder}/{name}.png')
	plt.show()