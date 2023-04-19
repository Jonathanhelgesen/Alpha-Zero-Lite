from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.optimizers import Adam
import tensorflow as tf
import tensorflow_model_optimization as tfmot
import numpy as np
import os
dirname = os.path.dirname(__file__)

class ANET:
    
	def __init__(self, model=None):
		self.model = model


	def make_model(self, board_size, neuron_counts=[64, 64], activation_func='relu', loss_func='categorical_crossentropy'):
		self.model = Sequential()
		input_shape = (board_size**2 + 1, 2)

		self.model.add(Flatten(input_shape=input_shape))

		for count in neuron_counts:
			self.model.add(Dense(count, activation=activation_func))

		self.model.add(Dense(board_size**2, activation='softmax'))

		optimizer = Adam(learning_rate=0.0001) # Var 0.0001
		self.model.compile(optimizer=optimizer, loss=loss_func, metrics=['accuracy'])
		self.convert_to_tflite()


	def one_hot_tuple(self, tuple):
		new_tuple = []
		for i in tuple:
			if i == 0:
				new_tuple.append([0, 0])
			elif i == 1:
				new_tuple.append([1, 0])
			elif i == 2:
				new_tuple.append([0, 1])
		return new_tuple


	def one_hot_tuple_list(self, lst):
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
	

	def get_tuples(self, data):
		tuples = []
		for d in data:
			flat_board = sum(d[0], [])
			player = d[1]
			flat_board.insert(0, player)
			tuple = flat_board
			tuples.append(self.one_hot_tuple(tuple))
		return tuples


	def fit(self, x, y, epochs=10, batch_size=1):
		tuples = self.get_tuples(x)
		x_processed = np.array(tuples)
		#x_processed = np.array(self.one_hot_tuple_list(tuples))
		y_processed = np.array(y)
		print(f'x shape: {x_processed.shape}')
		print(f'y shape: {y_processed.shape}')
		self.model.fit(x_processed, y_processed, epochs, batch_size)
		#self.convert_to_tflite()

	def predict(self, x):
		x_processed = np.array([self.one_hot_tuple(x)])
		#return self.predict_with_tflite(x)
		result = self.model(x_processed)
		#result = self.model.predict(x_processed, verbose=0)
		return result.numpy()

	def save(self, name):
		self.model.save(name)


	def convert_to_tflite(self):
		# Apply pruning
		"""
		pruning_params = {
            'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.50,
                                                                     final_sparsity=0.80,
                                                                     begin_step=0,
                                                                     end_step=1000)
        }
		model_for_pruning = tfmot.sparsity.keras.prune_low_magnitude(self.model, **pruning_params)

        # Apply quantization
		converter = tf.lite.TFLiteConverter.from_keras_model(model_for_pruning)
		converter.optimizations = [tf.lite.Optimize.DEFAULT]
		converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
		converter.inference_input_type = tf.int8
		converter.inference_output_type = tf.int8

		"""
    	# Convert the Keras model to a TensorFlow Lite model
		converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
		tflite_model = converter.convert()

    	# Save the TensorFlow Lite model to a file
		path = os.path.join(dirname, 'current_ANET.tflite')
		with open(path, 'wb') as f:
			f.write(tflite_model)


	def predict_with_tflite(self, x):
		x_processed = np.array(self.one_hot_tuple(x)).astype(np.float32)
		# Load the TensorFlow Lite model
		path = os.path.join(dirname, 'current_ANET.tflite')
		interpreter = tf.lite.Interpreter(model_path=path)
		interpreter.allocate_tensors()

		# Get input and output tensors
		input_details = interpreter.get_input_details()
		output_details = interpreter.get_output_details()

		# Make a prediction
		interpreter.set_tensor(input_details[0]['index'], x_processed)
		interpreter.invoke()
		output_data = interpreter.get_tensor(output_details[0]['index'])

		return output_data