from RL.Simulator import Simulator
import cProfile

def main():
	simulator = Simulator()
	simulator.run(10, 100, 64)



if __name__ == '__main__':
	main()
	#print('Timing program...')
	#cProfile.run('main()', sort='cumtime')