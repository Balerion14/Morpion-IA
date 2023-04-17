import multiprocessing
from random import randrange
from tqdm import tqdm
#from numba import jit, cuda
import threading
from multiprocessing import Manager
from multiprocessing.pool import Pool
from multiprocessing import Process
from multiprocessing import Lock
import os

import numpy as np
import itertools
from collections import deque
import pickle

import torch
from torch import nn
from minimax import create_minimax_player
play_minimax_move_randomized = create_minimax_player(True)
play_minimax_move_not_randomized = create_minimax_player(False)
from board import play_game, is_draw
from board import (CELL_X, CELL_O, RESULT_X_WINS, RESULT_O_WINS)

WIN_VALUE = 1.0
DRAW_VALUE = 1.0
LOSS_VALUE = 0.0

INPUT_SIZE = 9
OUTPUT_SIZE = 9


class TicTacNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.dl1 = nn.Linear(INPUT_SIZE, 36)#self.dl1 = nn.Linear(INPUT_SIZE, 36)
        self.dl2 = nn.Linear(36, 36)#self.dl2 = nn.Linear(36, 36)
        self.output_layer = nn.Linear(36, OUTPUT_SIZE)#self.output_layer = nn.Linear(36, OUTPUT_SIZE)

    def forward(self, x):
        x = self.dl1(x)
        x = torch.relu(x)

        x = self.dl2(x)
        x = torch.relu(x)

        x = self.output_layer(x)
        x = torch.sigmoid(x)
        return x

class NetContext:
    def __init__(self, policy_net, target_net, optimizer, loss_function):
        self.policy_net = policy_net

        self.target_net = target_net
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net = self.target_net.eval()

        self.optimizer = optimizer
        self.loss_function = loss_function


def create_qneural_player(net_context):
    def play(board):
        model = net_context.target_net
        return play_qneural_move(board, model)

    return play


def play_qneural_move(board, model):
    max_move_index, _ = select_valid_qneural_move(board, model)
    return board.play_move(max_move_index)


def select_valid_qneural_move(board, model):
    q_values = get_q_values(board, model)
    valid_move_indexes = board.get_valid_move_indexes()
    valid_q_values = get_valid_move_index_q_value_pairs(q_values,
                                                        valid_move_indexes)
    max_move_index, q_value = max(valid_q_values, key=lambda pair: pair[1])

    return max_move_index, q_value


def get_valid_move_index_q_value_pairs(q_values, valid_move_indexes):
    valid_q_values = []
    for vmi in valid_move_indexes:
        valid_q_values.append((vmi, q_values[vmi].item()))

    return valid_q_values


def get_q_values(board, model):
    inputs = convert_to_tensor(board)
    outputs = model(inputs)
    return outputs


def convert_to_tensor(board):
    return torch.tensor(board.board, dtype=torch.float)


"""def play_training_games_x(net_context, total_games=4000000,
                          discount_factor=1.0, epsilon=0.7, o_strategies=None):
    play_training_games(net_context, CELL_X, total_games, discount_factor,
                        epsilon, None, o_strategies)


def play_training_games_o(net_context, total_games=4000000,
                          discount_factor=1.0, epsilon=0.7, x_strategies=None):
    play_training_games(net_context, CELL_O, total_games, discount_factor,
                        epsilon, x_strategies, None)"""

#@jit(target_backend='cuda')
def play_training_games_x(net_context, total_games=1,#10000000
                         discount_factor=1.0, epsilon=0.7, o_strategies=None,modele_strategie_O=None):
    
    # Load weights from file
    lock = Lock()
    #load_net_context(net_context, 'weights.pt')
    # diviser le nombre total de partie par le nombre de processus disponible sur la machine et recuperer le resusltat dans une variable, recuperer egalement combien de fois je dois faire cce nombre de partie trouve pour arrive au nombre total de depart
    """calculer_parties(total_games)
    # recuperer les valeurs de retour de la fonction calculer_parties
    nb_partie_par_processus = calculer_parties(total_games)[0]
    multiplicateur = calculer_parties(total_games)[1] 

    
    # create the shared lock
    lock = Lock()
    processes = []
    # Creates 10 processes then starts them
    for i in range(20):
        p = multiprocessing.Process(target=play_training_games, args=(net_context, CELL_X, nb_partie_par_processus, discount_factor,epsilon, None, o_strategies,lock,modele_strategie_O))
        p.start()
        processes.append(p)

    # wait for all processes to finish
    for process in processes:
        process.join()"""

    play_training_games(net_context, CELL_X, total_games, discount_factor,
                        epsilon, None, o_strategies,lock,modele_strategie_O)

#@jit(target_backend='cuda')
def play_training_games_o(net_context, total_games=3000000,#10000000
                         discount_factor=1.0, epsilon=0.7, x_strategies=None):
    # Load weights from file
    #load_net_context(net_context, 'weights.pt')
    # diviser le nombre total de partie par le nombre de processus disponible sur la machine et recuperer le resusltat dans une variable, recuperer egalement combien de fois je dois faire cce nombre de partie trouve pour arrive au nombre total de depart
    calculer_parties(total_games)
    # recuperer les valeurs de retour de la fonction calculer_parties
    nb_partie_par_processus = calculer_parties(total_games)[0]
    multiplicateur = calculer_parties(total_games)[1] 

    # create the shared lock
    lock = Lock()
    processes = []
    # Creates 10 processes then starts them
    for i in range(20):
        p = multiprocessing.Process(target=play_training_games, args=(net_context, CELL_O, nb_partie_par_processus, discount_factor,epsilon,x_strategies,None,lock))
        p.start()
        processes.append(p)

    # wait for all processes to finish
    for process in processes:
        process.join()



    #play_training_games(net_context, CELL_O, total_games, discount_factor,
                        #epsilon, x_strategies, None)


def play_training_games(net_context, qplayer, total_games, discount_factor,
                        epsilon, x_strategies, o_strategies,lock,modele_strategie_O):
    if x_strategies:
        x_strategies_to_use = itertools.cycle([x_strategies(modele_strategie_O)])#[x_strategies(modele_strategie_O)]

    if o_strategies:
        o_strategies_to_use = itertools.cycle([o_strategies(modele_strategie_O)])#[o_strategies(modele_strategie_O)]

    for game in range(total_games):#tqdm
        move_history = deque()

        if not x_strategies:
            x = [create_training_player(net_context, move_history, epsilon)]
            x_strategies_to_use = itertools.cycle(x)

        if not o_strategies:
            o = [create_training_player(net_context, move_history, epsilon)]
            o_strategies_to_use = itertools.cycle(o)


        x_strategy_to_use = next(x_strategies_to_use)
        o_strategy_to_use = next(o_strategies_to_use)

        play_training_game(net_context, move_history, qplayer,
                           x_strategy_to_use, o_strategy_to_use,
                           discount_factor,lock)

        if (game+1) % (total_games / 10) == 0:
            epsilon = max(0, epsilon - 0.1)
            #print(f"{game+1}/{total_games} games, using epsilon={epsilon}...")


# Modifiez la fonction play_training_game() pour enregistrer les poids après chaque jeu.
def play_training_game(net_context, move_history, q_learning_player,
                       x_strategy, o_strategy, discount_factor,lock):
    board = play_game(x_strategy, o_strategy)#o_strategy

    # acquire the lock
    #with lock:
    update_training_gameover(lock,net_context, move_history, q_learning_player,
                             board, discount_factor)

    # Save weights to file
    #save_net_context(net_context, 'weights.pt')
    # Save net context object
    #save_net_context_object(net_context, 'net_context.pkl')

"""def play_training_game(net_context, move_history, q_learning_player,
                       x_strategy, o_strategy, discount_factor):
    board = play_game(x_strategy, o_strategy)

    update_training_gameover(net_context, move_history, q_learning_player,
                             board, discount_factor)"""


def update_training_gameover(lock,net_context, move_history, q_learning_player,
                             final_board, discount_factor):
    game_result_reward = get_game_result_value(q_learning_player, final_board)

    # move history is in reverse-chronological order - last to first
    next_position, move_index = move_history[0]

    backpropagate(lock,net_context, next_position, move_index, game_result_reward)

    for (position, move_index) in list(move_history)[1:]:
        with torch.no_grad():
            next_q_values = get_q_values(next_position, net_context.target_net)
            qv = torch.max(next_q_values).item()

        backpropagate(lock,net_context, position, move_index, discount_factor * qv)

        next_position = position

    with lock:
        net_context.target_net.load_state_dict(net_context.policy_net.state_dict())


def backpropagate(lock,net_context, position, move_index, target_value):
    #with lock :
    net_context.optimizer.zero_grad()
    output = net_context.policy_net(convert_to_tensor(position))

    target = output.clone().detach()
    target[move_index] = target_value
    illegal_move_indexes = position.get_illegal_move_indexes()
    for mi in illegal_move_indexes:
        target[mi] = LOSS_VALUE

    loss = net_context.loss_function(output, target)
    loss.backward()
    net_context.optimizer.step()


def create_training_player(net_context, move_history, epsilon):
    def play(board):
        model = net_context.policy_net
        move_index = choose_move_index(board, model, epsilon)
        move_history.appendleft((board, move_index))
        updated_board = board.play_move(move_index)

        return updated_board

    return play


def choose_move_index(board, model, epsilon):
    if epsilon > 0:
        random_value_from_0_to_1 = np.random.uniform()
        if random_value_from_0_to_1 < epsilon:
            return randrange(9)

    with torch.no_grad():
        q_values = get_q_values(board, model)
        max_move_index = torch.argmax(q_values).item()

    return max_move_index


def get_game_result_value(player, board):
    if is_win(player, board):
        return WIN_VALUE
    if is_loss(player, board):
        return LOSS_VALUE
    if is_draw(board):
        return DRAW_VALUE


def is_win(player, board):
    result = board.get_game_result()
    return ((player == CELL_O and result == RESULT_O_WINS)
            or (player == CELL_X and result == RESULT_X_WINS))


def is_loss(player, board):
    result = board.get_game_result()
    return ((player == CELL_O and result == RESULT_X_WINS)
            or (player == CELL_X and result == RESULT_O_WINS))

def save_net_context(net_context, filename):
    weights = net_context.policy_net.state_dict()
    torch.save(weights, filename)

def load_net_context(net_context, filename):
    try:
        weights = torch.load(filename)
        print(weights)
    except EOFError:
        weights = None
    if weights is not None:
        net_context.policy_net.load_state_dict(weights)

def save_net_context_object(net_context, filename):
    with open(filename, 'wb') as f:
        pickle.dump(net_context, f)

def calculer_parties(total_games):
    # Obtenir le nombre de processeurs disponibles sur la machine
    nb_cpu = 20
    # Calculer le nombre de partie par processeur
    games_par_cpu = total_games // nb_cpu
    # Calculer le nombre de fois que le nombre de partie par processeur doit être multiplié
    multiplication = total_games // (games_par_cpu * nb_cpu)
    return games_par_cpu, multiplication