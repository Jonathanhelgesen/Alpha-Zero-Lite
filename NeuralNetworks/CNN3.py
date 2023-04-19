import keras
from keras.layers import Input, Conv2D, Flatten, Dense, concatenate
from keras.models import Model
import tensorflow as tf
import tensorflow_model_optimization as tfmot
import math
import numpy as np
import os
dirname = os.path.dirname(__file__)


class CNN3:
    

	def __init__(self, model=None):
		self.model = model


	def make_model(self, board_size, layers, activation_func, loss_func):

		board_shape = (board_size, board_size, 23)
		input_board = Input(shape=board_shape, name='input_board')
		input_player = Input(shape=(2,), name='input_player')

		# Make convlutional layers
		conv1 = Conv2D(filters=128, kernel_size=3, activation='relu', padding='same')(input_board)
		conv2 = Conv2D(filters=128, kernel_size=3, activation='relu', padding='same')(conv1)
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
		self.convert_to_tflite()


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
	

	def create_feature_vector(self, board, player_id):
		pid = [1, 0] if player_id == 1 else [0, 1]
		n = len(board)
		feature_vector = [[[0 for _ in range(23)] for _ in range(n)] for _ in range(n)]
    
		for r in range(n):
			for c in range(n):
				if board[r][c] == 0:
					feature_vector[r][c][0] = 1
				elif board[r][c] == 1:
					feature_vector[r][c][1] = 1
				elif board[r][c] == 2:
					feature_vector[r][c][2] = 1
                
				# Top neighbor
				if r > 0:
					if board[r-1][c] == 0:
						feature_vector[r][c][3] = 1
					elif board[r-1][c] == 1:
						feature_vector[r][c][4] = 1
					elif board[r-1][c] == 2:
						feature_vector[r][c][5] = 1

				# Top right neighbor
				if r > 0 and c < n-1:
					if board[r-1][c+1] == 0:
						feature_vector[r][c][6] = 1
					elif board[r-1][c+1] == 1:
						feature_vector[r][c][7] = 1
					if board[r-1][c+1] == 2:
						feature_vector[r][c][8] = 1

				# Right neighbor
				if c < n-1:
					if board[r][c+1] == 0:
						feature_vector[r][c][9] = 1
					elif board[r][c+1] == 1:
						feature_vector[r][c][10] = 1
					elif board[r][c+1] == 2:
						feature_vector[r][c][11] = 1

				# Bottom neighbor
				if r < n-1:
					if board[r+1][c] == 0:
						feature_vector[r][c][12] = 1
					elif board[r+1][c] == 1:
						feature_vector[r][c][13] = 1
					elif board[r+1][c] == 2:
						feature_vector[r][c][14] = 1

				# Bottom left neighbor
				if r < n-1 and c > 0:
					if board[r+1][c-1] == 0:
						feature_vector[r][c][15] = 1
					elif board[r+1][c-1] == 1:
						feature_vector[r][c][16] = 1
					elif board[r+1][c-1] == 2:
						feature_vector[r][c][17] = 1

				# Left neighbor
				if c > 0:
					if board[r][c-1] == 0:
						feature_vector[r][c][18] = 1
					elif board[r][c-1] == 1:
						feature_vector[r][c][19] = 1
					elif board[r][c-1] == 2:
						feature_vector[r][c][20] = 1

				# Encode if cell is a goal cell
				if r == 0 or r == n-1:
					feature_vector[r][c][21] = 1
				if c == 0 or c == n-1:
					feature_vector[r][c][22] = 1
    
		return feature_vector, pid



	def unflatten_boards(self, board):
		row_len = int(math.sqrt(len(board)))
		return [board[i:i+row_len] for i in range(0, len(board), row_len)]
	

	def get_training_data(self, data):
		data = data
		training_data = []
		for d in data:
			board, pid = self.create_feature_vector(d[0], d[1])
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
		self.convert_to_tflite()


	def predict(self, board, player):
		x_board, x_pid = self.create_feature_vector(board, player)
		return self.predict_with_tflite(x_board, x_pid)
		return self.model([x_board, x_pid])[0]
		x_board, x_pid = self.one_hot_state(board, player)
	

	def save(self, name):
		self.model.save(name)
	

	def convert_to_tflite(self):
    	# Convert the Keras model to a TensorFlow Lite model
		converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
		tflite_model = converter.convert()

    	# Save the TensorFlow Lite model to a file
		path = os.path.join(dirname, 'current_ANET.tflite')
		with open(path, 'wb') as f:
			f.write(tflite_model)


	def predict_with_tflite(self, board, pid):
		#print(f'Board is looking like this: {board}')
		x_processed = [np.array([board]).astype(np.float32), np.array([pid]).astype(np.float32)]

		#print(f'x_processed type: {x_processed.dtype}')
		# Load the TensorFlow Lite model
		path = os.path.join(dirname, 'current_ANET.tflite')
		interpreter = tf.lite.Interpreter(model_path=path)
		interpreter.allocate_tensors()

		# Get input and output tensors
		input_details = interpreter.get_input_details()
		output_details = interpreter.get_output_details()

		# Make a prediction
		#interpreter.set_tensor(input_details[0]['index'], x_processed)
		interpreter.set_tensor(input_details[0]['index'], x_processed[0])
		interpreter.set_tensor(input_details[1]['index'], x_processed[1])
		interpreter.invoke()
		output_data = interpreter.get_tensor(output_details[0]['index'])

		return output_data