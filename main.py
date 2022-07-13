import numpy as np
import collections
import math, random
import time
from copy import deepcopy
from gym import *
from alphazero import *
from own_agent import *

def ChooseFirstPlayer():
    ''' Chooses the desired player for Player 1 '''
    player = input('Welcome to Connect 4. Key in your desired player for Player 1 (O):\n\
    1: Human Player\n\
    2. Monte Carlo Tree Search (1000 iterations)\n\
    3: Monte Carlo Tree Search (10000 iterations)\n\
    4: Minimax Agent with Expert Heuristic to Depth 2\n\
    5: AlphaZero Agent\n\
    6: Random Agent\n\
    7: Own Agent (Default: Random)\n')
    while True:
        if player in '1234567' and len(player) == 1:
            return int(player)
        else:
            player = input('Invalid choice. Please key in only a digit from 1 to 7.\n')

def ChooseSecondPlayer():
    ''' Chooses the desired player for Player 2 '''
    player = input('Welcome to Connect 4. Key in your desired player for Player 2 (X):\n\
    1: Human Player\n\
    2. Monte Carlo Tree Search (1000 iterations)\n\
    3: Monte Carlo Tree Search (10000 iterations)\n\
    4: Minimax Agent with Expert Heuristic to Depth 2\n\
    5: AlphaZero Agent\n\
    6: Random Agent\n\
    7: Own Agent (Default: Random)\n')
    while True:
        if player in '1234567' and len(player) == 1:
            return int(player)
        else:
            player = input('Invalid choice. Please key in only a digit from 1 to 7.\n')

def ChooseMove(moves):
    ''' Convert player moves in the form of grid cell (e.g. A4) to an index number '''

    movedict = {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6, 'g': 7}
    index = input('Key in cell number to play (i.e. A2) :\n')
    
    while True:
        row, col = -1, -1
        
        if index[0].isalpha() and index[1].isdigit():
            row = movedict.get(index[0].lower(), -1)
            col = int(index[1])
        if (row-1)*BOARDSIZE+col-1 in moves:
            return (row-1)*BOARDSIZE+col-1
        else:
            index = input('Invalid choice. Key in only playable cells:\n')

def ChooseTraining():
    ''' Choose whether to have training mode or testing mode '''
    mode = input('Choose the desired testing environment:\n1 for Var 1 (brick at D4), 2 for Var 2 (brick at E5), 3 for all possible brick squares\n')
    while True:
        if mode in '123' and len(mode) == 1:
            return int(mode)
        else:
            mode = input('Invalid choice. Please key in only the digits 1, 2 or 3.\n')

def ChooseNumGames():
    ''' Choose number of games '''
    num_games = input('Enter the number of games you wish to simulate/play:\n')
    while True:
        if num_games.isdigit() and int(num_games) > 0:
            return int(num_games)
        else:
            num_games = input('Invalid choice. Enter an integer larger than 0.\n')

    
def main():
    '''This runs the BTTT tournament'''
    # choose the hyperparameters
    firstplayer = ChooseFirstPlayer()
    secondplayer = ChooseSecondPlayer()
    train = ChooseTraining()
    num_games = ChooseNumGames()

    p1wins, p2wins, draw = 0, 0, 0

    for i in range(num_games):
        turn = 1   
        initialstate = np.zeros((BOARDSIZE, BOARDHEIGHT))
        
        if train == 1:
            initialstate[3, 3] = 3

        elif train == 2:
            initialstate[4, 4] = 3
            
        else:
            square = i%49
            initialstate[square//7,square%7] = 3
                
        env = gym(state = initialstate)
                
        while(True):
            print(f'Game {i+1}:\n')
            env.printboard()

            # if it is first player turn
            if(env.turn == 1):

                if firstplayer == 1:
                    action = ChooseMove(env.moves)
                elif firstplayer == 2:
                    action = mcts_agent(env, num_iter = 1000)
                elif firstplayer == 3:
                    action = mcts_agent(env, num_iter = 10000)
                elif firstplayer == 4:
                    action = minimax_agent(env, depth = 2)
                elif firstplayer == 5:
                    action = alphazero_agent(env)
                elif firstplayer == 6:
                    action = own_agent(env)
                else:
                    action = random_agent(env)

            # if it is second player turn
            else:
                if secondplayer == 1:
                    action = ChooseMove(env.moves)
                elif secondplayer == 2:
                    action = mcts_agent(env, num_iter = 1000)
                elif secondplayer == 3:
                    action = mcts_agent(env, num_iter = 10000)
                elif secondplayer == 4:
                    action = minimax_agent(env, depth = 2)
                elif secondplayer == 5:
                    action = alphazero_agent(env)
                elif secondplayer == 6:
                    action = own_agent(env)
                else:
                    action = random_agent(env)

            env.step(action)

            # check if anyone won the game
            if env.done:
                env.printboard()

                if env.reward == 0:
                    print('It is a draw!')
                elif env.reward == 1:
                    print('Player 1 (O) won!')
                else:
                    print('Player 2 (X) won!')
                break
            
        reward = env.reward
        if reward == 1:
            p1wins += 1
        elif reward == -1:
            p2wins += 1
        else:
            draw += 1
            
    print("In {} games, Player 1 wins {} times, Player 2 wins {} times, Draw {} times".format(num_games, p1wins, p2wins, draw))


if __name__ == '__main__':
    main()
