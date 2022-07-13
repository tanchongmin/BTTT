#!/usr/bin/env python
import numpy as np
import config

from shutil import copyfile
import random
from importlib import reload

from tensorflow.keras.utils import plot_model
import tensorflow as tf

from game import Game, GameState
from agent import Agent
from model import Residual_CNN
from funcs import playMatches, playMatchesBetweenVersions

import loggers as lg

from settings import run_folder, run_archive_folder
from tensorflow.keras.models import load_model
import pickle

from loss import softmax_cross_entropy_with_logits

def alphazero_agent(env = None):
    # load the best agent
    
    ## Change this to the agent you would like
#     MODEL_NAME = "AlphaZero Baseline.h5"
#     MODEL_NAME = "AlphaZero R2.h5"    
    MODEL_NAME = "AlphaZero R3.h5"

    
    game = Game(board = env.get_alphazero_state())

    ######## LOAD MODEL ########

    # create an untrained neural network objects from the config file
    best_NN = Residual_CNN(config.REG_CONST, config.LEARNING_RATE, (game.input_shape[0],) +  game.grid_shape, game.action_size, config.HIDDEN_CNN_LAYERS)

    # set weights to best model
    m_tmp = load_model(run_archive_folder + "models/" + MODEL_NAME, custom_objects={'softmax_cross_entropy_with_logits': softmax_cross_entropy_with_logits})
    best_NN.model.set_weights(m_tmp.get_weights())

    ######## CREATE THE PLAYERS ########

    current_player = Agent('current_player', game.state_size, game.action_size, config.MCTS_SIMS, config.CPUCT, best_NN)

    #### Run the MCTS algorithm and return an action
    action, pi, MCTS_value, NN_value = current_player.act(game.gameState, 0)

    return action