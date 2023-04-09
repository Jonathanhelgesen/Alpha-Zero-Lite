from keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, Add, Activation, Flatten, Dense, Concatenate
from keras.models import Model
import keras

class ResNet:

    def resnet_block(self, input_tensor, filters, kernel_size=(3, 3), strides=(1, 1), padding='same'):
        x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding)(input_tensor)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding)(x)
        x = BatchNormalization()(x)
        x = Add()([x, input_tensor])
        x = Activation('relu')(x)
        return x
    
    def __init__(self, board_size):
        board_input = Input(shape=(board_size, board_size, 2), name='board_input')
    
    	# Convolutional layers
        x = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')(board_input)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = self.resnet_block(x, filters=32)
        x = self.resnet_block(x, filters=32)
        x = self.resnet_block(x, filters=32)
    
        # Flatten and dense layers for the player identifier
        player_input = Input(shape=(2,), name='player_input')
        y = Dense(units=32, activation='relu')(player_input)
    
        # Concatenate the flattened board and dense player layers
        x = Flatten()(x)
        merged = Concatenate()([x, y])
    
        # Output layer with softmax activation
        output_layer = Dense(units=board_size**2, activation='softmax')(merged)
    
        # Define the model
        self.model = Model(inputs=[board_input, player_input], outputs=output_layer)
        
        optimizer = keras.optimizers.Adam(learning_rate=0.0001) # Should be added to params
        self.model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        

    def fit(self, x, y, epochs=10, batch_size=1):
        self.model.fit(x, y, epochs, batch_size)


    def predict(self, x_board, x_player):
        return self.model.predict([x_board, x_player])