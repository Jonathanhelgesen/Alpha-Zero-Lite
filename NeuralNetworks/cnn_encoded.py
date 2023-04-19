import keras
from keras.layers import Input, Conv2D, Flatten, Dense, Dropout
from keras.models import Model
import tensorflow as tf
import tensorflow_model_optimization as tfmot
import math
import numpy as np
import os
dirname = os.path.dirname(__file__)


class CNN_encoded:
    

	def __init__(self, model=None):
		self.model = model


	def make_model(self, board_size, layers, activation_func, loss_func):

		input_shape = (board_size, board_size, 11)
    
    	# Define input layer
		inputs = Input(shape=input_shape, name='input_layer')
    
    	# Define convolutional layers
		x = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(inputs)
		x = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(x)
    
    	# Define flatten layer
		x = Flatten()(x)
    
    	# Define dense layers
		x = Dense(units=512, activation='relu')(x)

		output = Dense(units=board_size**2, activation='softmax')(x)

		self.model = Model(inputs=inputs, outputs=output)

		# compile the model
		optimizer = keras.optimizers.Adam(learning_rate=0.001) # Should be added to params
		self.model.compile(optimizer=optimizer, loss=loss_func, metrics=['accuracy'])
		self.convert_to_tflite()


	def create_feature_vector(self, board, player_id):
		n = len(board)
		feature_vector = [[[0 for _ in range(11)] for _ in range(n)] for _ in range(n)]
    
		for r in range(n):
			for c in range(n):
				if board[r][c] == 0:
					feature_vector[r][c][0] = 1
					# Find number of different neigbors for empty cells
					neighbors = []
					for x, y in ((r-1,c), (r,c-1), (r+1,c), (r,c+1), (r-1,c+1), (r+1,c-1)):
						if 0 <= x < n and 0 <= y < n:
							neighbors.append(board[x][y])
            
					feature_vector[r][c][3] = neighbors.count(0)
					feature_vector[r][c][4] = neighbors.count(1)
					feature_vector[r][c][5] = neighbors.count(2)
				elif board[r][c] == 1:
					feature_vector[r][c][1] = 1
				elif board[r][c] == 2:
					feature_vector[r][c][2] = 1
                
				# Find number of neighbours for each cell
				if (r, c) in [(0, 0), (n-1, n-1)]:
					feature_vector[r][c][6] = 2
				elif (r, c) in [(0, n-1), (n-1, 0)]:
					feature_vector[r][c][6] = 3
				else:
					feature_vector[r][c][6] = 6

				# Encode if cell is a goal cell
				if r == 0 or r == n-1:
					feature_vector[r][c][7] = 1
				if c == 0 or c == n-1:
					feature_vector[r][c][8] = 1

				# Encode which player's turn it is
				feature_vector[r][c][9] = 1 if player_id == 1 else 0
				feature_vector[r][c][10] = 1 if player_id == 2 else 0
    
		return feature_vector



	def create_feature_vector2(self, board, player_id):
		n = len(board)
		feature_vector = [[[0 for _ in range(11)] for _ in range(n)] for _ in range(n)]
    
		for r in range(n):
			for c in range(n):
				if board[r][c] == 0:
					feature_vector[r][c][0] = 1
				elif board[r][c] == 1:
					feature_vector[r][c][1] = 1
				elif board[r][c] == 2:
					feature_vector[r][c][2] = 1

				if r == 0:
					feature_vector[r][c][3] = 1

                
				# Find number of neighbours for each cell
				if (r, c) in [(0, 0), (n-1, n-1)]:
					feature_vector[r][c][6] = 2
				elif (r, c) in [(0, n-1), (n-1, 0)]:
					feature_vector[r][c][6] = 3
				else:
					feature_vector[r][c][6] = 6

				# Encode if cell is a goal cell
				if r == 0 or r == n-1:
					feature_vector[r][c][7] = 1
				if c == 0 or c == n-1:
					feature_vector[r][c][8] = 1

				# Encode which player's turn it is
				feature_vector[r][c][9] = 1 if player_id == 1 else 0
				feature_vector[r][c][10] = 1 if player_id == 2 else 0
    
		return feature_vector

	

	def get_training_data(self, data):
		data = data
		training_data = []
		for d in data:
			vector = self.create_feature_vector(d[0], d[1])
			training_data.append(vector)
		return training_data



	def fit(self, x, y, epochs=10, batch_size=1):
		#x_boards, x_pids = self.get_training_data(x)
		x = self.get_training_data(x)
		x = np.array(x)
		self.model.fit(x, np.array(y), epochs, batch_size)
		self.convert_to_tflite()


	def predict(self, board, player):
		x = self.create_feature_vector(board, player)
		return self.predict_with_tflite(x)
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


	def predict_with_tflite(self, data):
		#print(f'Board is looking like this: {board}')

		#print(f'x_processed type: {x_processed.dtype}')
		# Load the TensorFlow Lite model
		data = np.array([data]).astype(np.float32)
		#print(f'Input shape is: {data.shape}')
		path = os.path.join(dirname, 'current_ANET.tflite')
		interpreter = tf.lite.Interpreter(model_path=path)
		interpreter.allocate_tensors()

		# Get input and output tensors
		input_details = interpreter.get_input_details()
		output_details = interpreter.get_output_details()

		#print(f'Input details: {input_details}')

		# Make a prediction
		#interpreter.set_tensor(input_details[0]['index'], x_processed)
		interpreter.set_tensor(input_details[0]['index'], data)
		interpreter.invoke()
		output_data = interpreter.get_tensor(output_details[0]['index'])

		return output_data