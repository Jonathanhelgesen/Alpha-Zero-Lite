from keras.optimizers import Adagrad, SGD, RMSprop, Adam
from enum import Enum

learning_rate = 0.001

optimizers = {
    'adam': Adam(learning_rate=learning_rate),
    'sgd': SGD(learning_rate=learning_rate),
    'rmsprop': RMSprop(learning_rate=learning_rate),
    'adagrad': Adagrad(learning_rate=learning_rate)
}

activation_funcs = ['relu', 'tanh', 'sigmoid', 'linear']

params = {
    'board_size': 5,
    'num_rollouts': 6400,
    'time_limit': 1,
    'epsilon': 0.3, 	# For epsilon-greedy rollouts. 1 is completely random, 0 is only ANET
    'num_episodes': 1, 	# How many episodes are played before fetching cases from RBUF and training ANET
    'C': 1.4,
    'save_interval': 5,
    'num_anets': 10,
    
	# NN params
	'layers': [64],
    'optimizer': optimizers['adam'],
    'activation_func': activation_funcs[0],
    'loss_func': 'categorical_crossentropy',
    
	# RBUF params
	'max_size': 1024,
    'batch_size': 64,
    
	# TOPP params
	'anet_folder': 'saved_nets_demo',
    'anets': ['init_cnn.h5', 'anet1.h5', 'anet2.h5', 'anet3.h5', 'anet4.h5'],
    'num_matches': 50,
    
	# Visualization
	'verbose': False,
    'show_time': None
}