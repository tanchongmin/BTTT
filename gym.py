import numpy as np
import collections
import math, random
import time
from copy import deepcopy

### Define parameters here
BOARDSIZE = 7
BOARDHEIGHT = 7
WINLENGTH = 4
        
class gym:
    def __init__(self, **kwargs):
        '''
        Data structure to represent state of the environment
        self.state : state of board
        '''
        # default settings
        self.boardsize = kwargs.get('boardsize', BOARDSIZE)
        self.boardheight = kwargs.get('boardheight', BOARDHEIGHT)
        self.mapping = {0: ' ', 1: 'O', 2: 'X', 3: 'B'}
        self.display = kwargs.get('display', 'tensor')
        self.state = kwargs.get('state', np.zeros((self.boardheight, self.boardsize), dtype = int))
        self.initialstate = deepcopy(self.state)
        self.winlength = kwargs.get('winlength', WINLENGTH)
        self.turn = kwargs.get('turn', 1)
        self.reward = kwargs.get('reward', 0)
        self.done = kwargs.get('done', False)
        self.moves = [i*self.boardsize+j for i in range(0, self.boardheight) for j in range(0, self.boardsize) if self.state[i, j] == 0]
        
    def place(self, grid, stone):
        '''
        Make a move at (grid[0], grid[1]) without changing the environment
        '''
        self.state[grid[0], grid[1]] = stone
        self.moves.remove(grid[0]*self.boardsize+grid[1])
        
    def step(self, action):
        '''
        Takes action at self.state and returns the next step
        '''
        # check to see if action can be taken

        if(action not in self.moves):
            print('Invalid move')
            return
            
        # remove current action
        self.moves.remove(action)

        # play the move at the row and col
        h, w = action//self.boardsize, action%self.boardsize
        self.state[h, w] = self.turn
    
        # check if anyone won based on just that move
        win = False
        for shift in range(4):
            # Vertical
            if self.checkwin(h+shift-(self.winlength-1),w, 1,0):
                win = True
            # Horizontal
            if self.checkwin(h,w+shift-(self.winlength-1), 0,1):
                win = True
            # Top Right Diagonal
            if self.checkwin(h+shift-(self.winlength-1),w+shift-(self.winlength-1), 1,1):
                win = True
            # Top Left Diagonal
            if self.checkwin(h+shift-(self.winlength-1),w+(self.winlength-1)-shift, 1,-1):
                win = True

        if win > 0:
            self.done = True
            self.reward = 1 if self.turn == 1 else -1

        # if no available moves left, then it is a draw
        if self.moves == []:
            self.done = True

        # update turn
        self.turn = 1 if self.turn == 2 else 2
        
    def sample(self):
        return(np.random.choice(self.moves))

    def doubleStep(self, action, minimax_depth = 2, p = 0.5):
        # Initialize state
        self.step(action)

        # if not ended yet, chance% probability to play a game with a minimax agent
        # otherwise, random action
        
        if not self.done:
            if np.random.rand() < p:
                action = minimax_agent(self, depth = minimax_depth)
                self.step(action)
            else:
                self.step(self.sample())

    def checkwin(self, h, w, dh, dw):
        # check if boundary case
        if(max(w, w+(self.winlength-1)*dw) > self.boardsize - 1):
            return 0
        if(max(h, h+(self.winlength-1)*dh) > self.boardheight - 1):
            return 0
        if(min(h, h+(self.winlength-1)*dh, w, w+(self.winlength-1)*dw) < 0):
            return 0

        # check if player 1 wins
        gamevalue = 1
        for i in range(0, self.winlength):
            gamevalue *= self.state[h + i*dh, w + i*dw]
        if gamevalue == 1:
            return 1
        if gamevalue == 2**self.winlength:
            return 2

    def reset(self):
        self.state = deepcopy(self.initialstate)
        self.moves = [i*self.boardsize+j for i in range(0, self.boardheight) for j in range(0, self.boardsize) if self.state[i, j] == 0]
        self.turn = 1
        self.reward = 0
        self.done = False

    def printboard(self):
        rowheading = "ABCDEFG"
        rowindex = 0
        
        for row in self.state:
            print(rowheading[rowindex], end = '')
            rowindex += 1
            
            print('|', end = '')
            for index in row:
                print(self.mapping[index], end = '|')
            print()
            
        # bottom border
        print(' ', end = '')
        print('__'*(self.boardsize+1))

        # bottom index
        print(' ', end = '')
        print('|', end = '')
        for num in range(self.boardsize):
            print(num+1, end = '|')
        print()
        
    def get_tensor_state(self):
        ''' This returns the state in tensor form for deep learning methods '''
        tensor = np.zeros((4, self.boardheight, self.boardsize))
        for i in range(self.boardheight):
            for j in range(self.boardsize):
                # first 2D grid is for turn
                tensor [0, i, j] = 1 if self.turn == 1 else -1
                # next 2D grid is for O
                if self.state[i, j] == 1:
                    tensor[1, i, j] = 1
                # next 2D grid is for X
                elif self.state[i, j] == 2:
                    tensor[2, i, j] = 1
                # next 2D grid is for B
                elif self.state[i, j] == 3:
                    tensor[3, i, j] = 1
        return tensor
    
    def get_alphazero_state(self):
        ''' This returns the state for the alphazero method to use '''
        newstate = self.state.flatten()
        for i in range(len(newstate)):
            if newstate[i]==1:
                newstate[i] = [1,-1][self.turn-1]
            elif newstate[i]==2:
                newstate[i] = [-1,1][self.turn-1]
            elif newstate[i]==3:
                newstate[i]=100
            
        return newstate
        
    def get_state(self):
        return self.state, self.turn, self.reward, self.done
        
    def get_env(self):
        return gym(state = deepcopy(self.state), turn = self.turn, reward = self.reward, done = self.done)
    
        
#### Monte Carlo Agent here (default agent)

### Define parameters here
random_rollout_iter = 1
exploration_param = 1

def randomPolicy(myenv):
    '''
    Random rollouts in the environment
    '''
    
    # if terminal state, return reward
    if myenv.done == True:
        return myenv.reward

    # else play until terminal state, return its final reward
    reward = 0

    # do random rollouts for random_rollout_iter number of times
    for _ in range(random_rollout_iter):
        # create a copy of the environment
        env = myenv.get_env()
        
        while not env.done:
            env.step(env.sample())

        reward += env.reward
    
    return reward/random_rollout_iter
    
def heuristicPolicy(env):
    '''
    Heuristic value approximation using the expert heuristic instead of rollouts
    '''
    
    # if terminal state, return reward
    if env.done == True:
        return env.reward
        
    # else do heuristic evaluation (no need random rollouts)
    else:
        value = evaluate(env.state)
#        print(value)
#        print((np.log10(abs(value)+1e-6)))
#        print((4+np.log10(abs(value)+1e-6))/4 * np.sign(value))
        return (6+np.log10(abs(value)+1e-6))/6 * np.sign(value)

class Node:
    def __init__(self, parent, env):
        '''
        Data structure for a node of the MCTS tree
        self.env : gym environment represented by the node
        self.parent : Parent of the node in the MCTS tree
        self.numVisits : Number of times the node has been visited
        self.totalReward : Sum of all rewards backpropagated to the node
        self.allChildrenAdded : Denotes whether all actions from the node have been explored
        self.children : Set of children of the node in the MCTS tree
        '''

        # state parameters
        self.env = env

        # parameters for the rest
        self.parent = parent
        self.numVisits = 0
        self.totalReward = 0

        # if the state is a terminal state, by default all children are added
        self.allChildrenAdded = self.env.done
        self.children = {}
        # children of the form "action: Node()"
        
class MonteCarloTreeSearch:
    def __init__(self, num_iter, explorationParam, playoutPolicy=randomPolicy, random_seed=None):
        '''
        self.num_iter : Number of iterations to play out
        self.explorationParam : exploration constant used in computing value of node
        self.playoutPolicy : Policy followed by agent to simulate rollout from leaf node
        self.root : root node of MCTS tree
        '''
        self.num_iter = num_iter
        self.explorationParam = explorationParam
#        self.playoutPolicy = playoutPolicy
        self.playoutPolicy = heuristicPolicy
        self.root = None

    def buildTreeAndReturnBestAction(self, env):
        '''
        Function to build MCTS tree and return best action at initialState
        '''
        self.root = Node(parent=None, env = env)

#        timeout_start = time.time()
#        numiter = 0
#
#        while time.time() < timeout_start + self.timeout:
#            self.addNodeAndBackpropagate()
#            numiter += 1

#        print('Numiter:', numiter)

        for i in range(self.num_iter):
            self.addNodeAndBackpropagate()

        # return action with highest value
   
        values = np.full(self.root.env.state.shape[0]*self.root.env.state.shape[1], -2, dtype = float)
        numvisits = np.full(self.root.env.state.shape[0]*self.root.env.state.shape[1], -2, dtype = int)

        for action, cur_node in self.root.children.items():
            values[action] = (cur_node.totalReward/cur_node.numVisits)
            numvisits[action] = cur_node.numVisits

#         print('Values')
#         for i in range(BOARDHEIGHT):
#             print(np.round(values[i*BOARDSIZE: (i+1)*BOARDSIZE], 3))
#         print ('Num visits')
#         for i in range(BOARDHEIGHT):
#             print(np.round(numvisits[i*BOARDSIZE: (i+1)*BOARDSIZE], 0))

#        return np.argmax(values)
        return np.argmax(numvisits)
        
    def addNodeAndBackpropagate(self):
        '''
        Function to run a single MCTS iteration
        '''
        node = self.addNode()
        reward = self.playoutPolicy(node.env)
        self.backpropagate(node, reward)

    def addNode(self):
        '''
        Function to add a node to the MCTS tree
        '''
        cur_node = self.root
        while not cur_node.env.done:
        # this is to check if the current node is a leaf node
            if cur_node.allChildrenAdded:
                cur_node = self.chooseBestActionNode(cur_node, self.explorationParam)
            else:
                actions = cur_node.env.moves
                for action in actions:
                    if action not in cur_node.children:
                        new_env = cur_node.env.get_env()
                        new_env.step(action)
                        newNode = Node(parent=cur_node, env = new_env)
                        cur_node.children[action] = newNode
                        if len(actions) == len(cur_node.children):
                            cur_node.allChildrenAdded = True
                        return newNode
        return cur_node
                
    def backpropagate(self, node, reward):
        '''
        This function implements the backpropation step of MCTS.
        '''
        # adds rewards to all nodes along the path (upwards to parent)
        while (node is not None):

            if node.env.turn == 1:
                node.totalReward -= reward
            else:
                node.totalReward += reward

            # add counter to node
            node.numVisits += 1
            # go up one level
            node = node.parent
        

    def chooseBestActionNode(self, node, explorationValue):
        bestValue = float("-inf")
        bestNodes = []
        
        for action, child in enumerate(node.children.values()):
            
            '''
            Populate the list bestNodes with all children having maximum value
                       
            Value of all nodes should be computed as mentioned in question 3(b).
            All the nodes that have the largest value should be included in the list bestNodes.
            We will then choose one of the nodes in this list at random as the best action node.
            '''
            
            value = child.totalReward/child.numVisits + explorationValue * math.sqrt(math.log(node.numVisits)/child.numVisits)

            # if this value is higher, set this to be new bestValue, bestNode will be current child node
            if value > bestValue:
                bestValue = value
                bestNodes = [child]
            
            # if the value ties with current best value, then current child node will be appended to the list of best nodes
            elif value == bestValue:
                bestNodes.append(child)
            
        return np.random.choice(bestNodes)

    ### MCTS agent here ###
        
def mcts_agent(env = None, num_iter = 2500):
    '''
    Input: environment

    Output: MCTS action
    '''
    
    # hard-set this
    num_iter = 2500
    
    # create a local copy
    env = env.get_env()
    
    # Do the Monte Carlo Policy prediction for this step, without changing the environment
    mcts = MonteCarloTreeSearch(num_iter = num_iter, explorationParam=exploration_param, random_seed=42)
    action = mcts.buildTreeAndReturnBestAction(env)
    
    return action
    
    ### Random agent here ###
    
def random_agent(env = None):
    '''
    Input: environment
    Output: random action
    '''
    
    return env.sample()
    
    ### Minimax agent here ###
    
def minimax_agent(env = None, depth = 2):
    '''
    Input: environment
    Output: Minimax action
    '''

    env = env.get_env()
    
    if env.moves == []:
        print("No moves to choose from in environment")
        return None
        
    maxreward = -10000 if env.turn == 1 else 10000
    bestaction = []
    
    for action in env.moves:
        new_env = env.get_env()
        new_env.step(action)
        reward = minimax(new_env, alpha = -10000, beta = 10000, depth = depth-1)
        if reward == maxreward:
            bestaction.append(action)
        if (env.turn == 1 and reward > maxreward) or (env.turn == 2 and reward < maxreward):
            maxreward = reward
            bestaction = [action]

    # this is a fail-safe in case minimax does not return any action
    if bestaction == []:
        print("No best action, choosing the first action instead")
        return env.moves[0]
    
    return np.random.choice(bestaction)
    # return bestaction[0]
        
def minimax(env, alpha, beta, depth = 2):
    if (env.done):
        return env.reward
    
    if(depth == 0):
        return evaluate(env.state)

    # alpha beta pruning
    if alpha >= beta:
        if env.turn == 1:
            return alpha
        else:
            return beta
        
    maxreward = -10000 if env.turn == 1 else 10000
    
    for action in env.moves:
        new_env = env.get_env()
        new_env.step(action)
#        print('Player {}: reward {}'.format(state.turn, reward))
        reward = minimax(new_env, alpha = alpha, beta = beta, depth = depth-1)
        if (env.turn == 1 and reward >= maxreward) or (env.turn == 2 and reward <= maxreward):
            maxreward = reward
            
        # update alpha and beta
        if env.turn == 1 and maxreward > alpha:
            alpha = maxreward
        if env.turn == 2 and maxreward < beta:
            beta = maxreward
            
    return maxreward
    
### Helper function to evaluate the state of a board position heuristically
def evaluate(state):
    ''' This returns the value of the current board position'''
    
    statemap = {0: ".", 1: "O", 2: "X", 3: "B"}
    
    value = 0
    # do horizontals
    for i in range(state.shape[0]):
        for j in range(state.shape[1]):
            row = ""
            col = ""
            diag = ""
            reversediag = ""
            for offset in range(WINLENGTH):
                if(j + offset < state.shape[1]):
                    row += statemap[state[i, j+offset]]
                else:
                    row += "B"
                    
                if(i + offset < state.shape[0]):
                    col += statemap[state[i+offset, j]]
                else:
                    col += "B"
                    
                if(i + offset < state.shape[0] and j + offset < state.shape[1]):
                    diag += statemap[state[i+offset, j+offset]]
                else:
                    diag += "B"
                    
                if(i + offset < state.shape[0] and j - offset >= 0):
                    reversediag += statemap[state[i+offset, j-offset]]
                else:
                    reversediag += "B"
                
            value += computereward(row)
            value += computereward(col)
            value += computereward(diag)
            value += computereward(reversediag)
            
    return value
            
def computereward(curstring):
    ''' Every space decreases points by 100
        If there is a block, 0 pts
        If there are both X and O, 0 pts
        If there is a space at the front and space at the end, x2 points
        O has 1.5x the points of X to encourage X to block
        
        Sample points for 4-in-a-row
        X... ...X -0.000001 pts
        .X.. ..X. -0.000002 pts
        X.X. X..X .X.X XX.. ..XX -0.0001 pts
        .XX. -0.0002 pts
        X.XX XX.X XXX. .XXX -0.01 pts
        XXXX -1 pt
        O... ...O 0.0000015 pts
        .O.. ..O. 0.000003 pts
        O.O. O..O .O.O OO.. ..OO 0.00015 pts
        .OO. 0.0003 pts
        O.OO OO.O OOO. .OOO 0.015 pts
        OOOO 1.5 pt
    '''
    
    p1stone = 0
    p2stone = 0
    blank = 0
    reward = 1

    # we give higher score to positions that are centralized
    bonus = 2 if (curstring[0]+curstring[-1]=="..") else 1
    
    for each in curstring:
        # ignore anything with a brick
        if each == "B":
            return 0
            
        elif each == "O":
            p1stone += 1
                
        elif each == "X":
            p2stone += 1
            
        else:
            blank += 1

    # ignore positions if there are both O and X
    if p1stone > 0 and p2stone > 0:
        return 0
        
    elif p1stone > 0:
        return bonus*reward*1.5/(100**blank)
        
    elif p2stone > 0:
        return -bonus*reward/(100**blank)
        
    else:
        return 0


