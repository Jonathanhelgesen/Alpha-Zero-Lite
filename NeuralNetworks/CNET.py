from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.optimizers import Adam
import numpy as np

class CNET:
    
	def __init__(self, board_size, neuron_counts, activation_func, loss_func):
		self.model = Sequential()
		input_shape = (board_size**2 + 1, 2)

		self.model.add(Flatten(input_shape=input_shape))

		for count in neuron_counts:
			self.model.add(Dense(count, activation=activation_func))

		self.model.add(Dense(1, activation='sigmoid'))

		optimizer = Adam(learning_rate=0.0001) # Var 0.0001
		self.model.compile(optimizer=optimizer, loss=loss_func, metrics=['accuracy'])


	def one_hot_tuple(self, tuple):
		new_tuple = []
		for i in tuple:
			if i == 0:
				new_tuple.append([0, 0])
			elif i == 1:
				new_tuple.append([1, 0])
			elif i == 2:
				new_tuple.append([0, 1])
		return [new_tuple]


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


	def fit(self, x, y, epochs=10, batch_size=1):
		x_processed = np.array(self.one_hot_tuple(x))
		y_processed = np.array(y)
		self.model.fit(x_processed, y_processed, epochs, batch_size)

	def predict(self, x):
		x_processed = np.array(self.one_hot_tuple(x))
		return self.model.predict(x_processed, verbose=0)
	


if __name__ == '__main__':
	layers = [128, 128, 128]
	anet = CNET(5, layers, 'softmax', 'adam', 'categorical_crossentropy')
	anet.fit()