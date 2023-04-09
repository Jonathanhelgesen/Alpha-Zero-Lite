import keras
from keras.layers import Input, Conv2D, Flatten, Dense, concatenate
from keras.models import Model

# define the shape of the board state
input_shape = (5, 5, 2)

# create two input layers for the board state and player to move
input_board = Input(shape=input_shape, name='input_board')
input_player = Input(shape=(2,), name='input_player')

# create the convolutional layers for the board state
conv1 = Conv2D(filters=48, kernel_size=3, activation='relu')(input_board) # Var 32
conv2 = Conv2D(filters=96, kernel_size=3, activation='relu')(conv1) # Var 64

# flatten the output from the convolutional layers
flatten = Flatten()(conv2)

# concatenate the flattened board state and player to move
concat = concatenate([flatten, input_player])

# add a dense layer
dense = Dense(units=160, activation='relu')(concat) # Var 64

# add the output layer
output = Dense(units=5*5, activation='softmax')(dense)

# create the model with multiple inputs
model = Model(inputs=[input_board, input_player], outputs=output)

# compile the model
optimizer = keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

class CNN:
    
	def __init__(self, board_size, layers, activation_func, loss_func):

		board_shape = (board_size, board_size, 2)
		input_board = Input(shape=board_shape, name='input_board')
		input_player = Input(shape=(2,), name='input_player')

		# Make convlutional layers
		conv1 = Conv2D(filters=64, kernel_size=3, activation='relu')(input_board) # Var 32
		conv2 = Conv2D(filters=128, kernel_size=3, activation='relu')(conv1)
		flatten = Flatten()(conv2)
		concat = concatenate([flatten, input_player])

		# Add dense layers
		x = concat
		for i in layers:
			x = Dense(i, activation='relu')(x)

		output = Dense(units=board_size**2, activation='softmax')(x)

		self.model = Model(inputs=[input_board, input_player], outputs=output)

		# compile the model
		optimizer = keras.optimizers.Adam(learning_rate=0.0001) # Should be added to params
		self.model.compile(optimizer=optimizer, loss=loss_func, metrics=['accuracy'])


	def fit(self, x, y, epochs=10, batch_size=1):
		self.model.fit(x, y, epochs, batch_size)


	def predict(self, x_board, x_player):
		return self.model.predict([x_board, x_player])