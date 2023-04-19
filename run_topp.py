from TOPP.tournament import Tournament

"""
anets = ['anet_0', 'anet_1', 'anet_2', 'anet_3', 'anet_5', 'anet_10',
         'anet_15', 'anet_20', 'cnn_0', 'cnn_1', 'cnn_2', 'cnn_3', 'cnn_4', 'cnn_5',
         'cnn_6', 'cnn_7', 'cnn_8', 'cnn_9', 'cnn_10', 'cnn_11', 'cnn_12', 'cnn_13',
         'cnn_14', 'cnn_15']
"""


anets = ['ann_4x4_0', '4cnn_1', '4cnn_3', '4cnn_5', '4cnn_7','4cnn_9', 
         '4cnn_11', '4cnn_13', '4cnn_15', '4cnn_17', '4cnn_20' ]
         #'4cnn_5', '4cnn_10', 'cnn_15']


t = Tournament(11, 1000, 4)
t.run(anets)