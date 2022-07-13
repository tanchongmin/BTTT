import numpy as np
from time import perf_counter
from gym import *
from alphazero import *
from own_agent import *

def main():

    agent = ["Own Agent (Default: Random)", "Random", "MCTS 1000", "MCTS 10000", "Minimax (Depth 2)", "AlphaZero"]

    initialstate = np.zeros((BOARDSIZE, BOARDHEIGHT))
    
    initialstate[3, 3] = 3
    env = gym(state = initialstate)
    
    for firstplayer in range(6):
        starttime = perf_counter()
        for i in range(100):
            if firstplayer == 0:
                action = own_agent(env)
            elif firstplayer == 1:
                action = random_agent(env)
            elif firstplayer == 2:
                action = mcts_agent(env, num_iter = 1000)
            elif firstplayer == 3:
                action = mcts_agent(env, num_iter = 10000)
            elif firstplayer == 4:
                action = minimax_agent(env, depth = 2)
            elif firstplayer == 5:
                action = alphazero_agent(env)

        endtime = perf_counter()
        print(agent[firstplayer], 'took', (endtime-starttime)/100, 'seconds')


if __name__ == '__main__':
    main()
