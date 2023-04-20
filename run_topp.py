from TOPP.tournament import Tournament
from params import params


anets = []

for anet in params['anets']:
    folder = params['anet_folder']
    anets.append(f'simulations\\{folder}\\{anet}')


t = Tournament(len(anets), params['num_matches'], params['board_size'])
t.run(anets)