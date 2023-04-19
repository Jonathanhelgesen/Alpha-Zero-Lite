import keras
from keras.layers import Input, Conv2D, Flatten, Dense, concatenate, BatchNormalization, MaxPooling2D
from keras.models import Model
import tensorflow as tf
import tensorflow_model_optimization as tfmot
import math
import numpy as np
import os
dirname = os.path.dirname(__file__)


class CNN_plain:
    

	def __init__(self, model=None, name='Margot'):
		self.model = model
		self.name = name


	def make_model(self, board_size, layers, activation_func, loss_func):

		board_shape = (board_size, board_size, 3)
		input_board = Input(shape=board_shape, name='input_board')
		input_player = Input(shape=(2,), name='input_player')

		# Make convlutional layers
		conv1 = Conv2D(filters=128, kernel_size=3, activation='relu', padding='same')(input_board)
		#conv1 = BatchNormalization()(conv1)
		#conv1 = MaxPooling2D(pool_size=(2, 2))(conv1)
		conv2 = Conv2D(filters=64, kernel_size=3, activation='relu', padding='same')(conv1)
		#conv2 = BatchNormalization()(conv2)
		#conv2 = MaxPooling2D(pool_size=(2, 2))(conv2)
		flatten = Flatten()(conv2)
		concat = concatenate([flatten, input_player])

		# Add dense layers
		x = concat
		for i in layers:
			x = Dense(i, activation=activation_func)(x)

		output = Dense(units=board_size**2, activation='softmax')(x)

		self.model = Model(inputs=[input_board, input_player], outputs=output)

		# compile the model
		optimizer = keras.optimizers.Adam(learning_rate=0.001) # Should be added to params
		self.model.compile(optimizer=optimizer, loss=loss_func, metrics=['accuracy'])


	def get_board_and_pid(self, board, player):
		# Encode the board as a one-hot matrix
		player_id = [1, 0] if player == 1 else [0, 1]
		n = len(board)
		board_one_hot = np.zeros((n, n, 3))
		for i in range(n):
			for j in range(n):
				if board[i][j] == 0:
					board_one_hot[i][j][0] = 1
				elif board[i][j] == 1:
					board_one_hot[i][j][1] = 1
				elif board[i][j] == 2:
					board_one_hot[i][j][2] = 1
		return np.array([board_one_hot], dtype=np.float32), np.array([player_id], dtype=np.float32)


	def one_hot_state(self, board, player):
		encoded_player = [1, 0] if player == 1 else [0, 1]

		values = [0, 1, 2]
		encoded_board = []

		for row in board:
			encoded_row = []
			for value in row:
				one_hot = [0] * len(values)
				one_hot[values.index(value)] = 1
				encoded_row.append(one_hot)
			encoded_board.append(encoded_row)

		return encoded_board, encoded_player



	def unflatten_boards(self, board):
		row_len = int(math.sqrt(len(board)))
		return [board[i:i+row_len] for i in range(0, len(board), row_len)]
	

	def get_training_data(self, data):
		data = data
		training_data = []
		for d in data:
			board, pid = self.one_hot_state(d[0], d[1])
			training_data.append([board, pid])
		return training_data
	

	def list_to_numpy(self, data):
		x1 = np.array([x[0] for x in data])  # Input 1 as numpy array
		x2 = np.array([x[1] for x in data])  # Input 2 as numpy array
		return [x1, x2]


	def fit(self, x, y, epochs=10, batch_size=1):
		#x_boards, x_pids = self.get_training_data(x)
		x = self.get_training_data(x)
		x = self.list_to_numpy(x)
		self.model.fit(x, np.array(y), epochs, batch_size)


	def predict(self, board, player):
		x_board, x_pid = self.get_board_and_pid(board, player)
		#return self.predict_with_tflite(x_board, x_pid)
		result = self.model([x_board, x_pid])
		return result.numpy()


	def test(self, x, y, epochs=10, batch_size=1):
		#x_boards, x_pids = self.get_training_data(x)
		x = self.get_training_data(x)
		x = self.list_to_numpy(x)
		loss, accuracy = self.model.evaluate(x, np.array(y), epochs, batch_size)
		print('Test loss:', loss)
		print('Test accuracy:', accuracy)
	

	def save(self, name):
		self.model.save(name)