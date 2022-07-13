import numpy as np
import collections
import math, random, os

import time
from copy import deepcopy
from gym import *

def own_agent(env = None):
    '''
    This is the code for your own agent for BTTT to test against the default Minimax, MCTS, DQN, random agents
    Defaults as a random agent
    Edit the code to pit your agent against the other agents in the BTTT tournament using `main.py`
    
    Input: environment state

    Output: Action
    '''
    return env.sample()
    