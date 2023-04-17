import multiprocessing
import numpy as np
import torch
#import numba
from torch.nn import MSELoss
from multiprocessing import Manager
from multiprocessing.pool import Pool
from multiprocessing import Process
from multiprocessing import Lock

from board import play_random_move, play_games, Board, play_human_move,CELL_X,CELL_O
from minimax import create_minimax_player, play_minimax_move
from qneural import (TicTacNet, NetContext, create_qneural_player,
                            get_q_values, play_training_games_x,
                            play_training_games_o, play_training_games, calculer_parties,play_qneural_move)

play_minimax_move_randomized = create_minimax_player(True)
play_minimax_move_not_randomized = create_minimax_player(False)

if __name__ == '__main__':

    policy_net = TicTacNet()
    target_net = TicTacNet()
    sgd = torch.optim.SGD(policy_net.parameters(), lr=0.1)
    loss = MSELoss()
    net_context = NetContext(policy_net, target_net, sgd, loss)

    # 2eme ia
    policy_net2 = TicTacNet()
    target_net2 = TicTacNet()
    sgd2 = torch.optim.SGD(policy_net2.parameters(), lr=0.1)
    loss2 = MSELoss()
    net_context2 = NetContext(policy_net2, target_net2, sgd2, loss2)



    #---------------------------------------------------
    """total_games = 2000000
    calculer_parties(total_games)
    # recuperer les valeurs de retour de la fonction calculer_parties
    nb_partie_par_processus = calculer_parties(total_games)[0]
    multiplicateur = calculer_parties(total_games)[1] 

    
    # create the shared lock
    lock = Lock()
    processes = []
    # Creates 10 processes then starts them
    for i in range(20):
        p = multiprocessing.Process(target=play_training_games, args=(net_context, CELL_X, nb_partie_par_processus, 1.0,0.7, None, [play_minimax_move],lock))
        p.start()
        processes.append(p)

    # wait for all processes to finish
    for process in processes:
        process.join()"""
    
    #---------------------------------------------------

    """with torch.no_grad():
        board = Board(np.array([1, -1, -1, 0, 1, 1, 0, 0, -1,]))#, 0, 0, 0, 0, 0, 0, 0]))
        q_values = get_q_values(board, net_context.target_net)
        print(f"Before training q_values = {q_values}")"""

    print("Training qlearning X vs. random...")
    play_training_games_x(net_context=net_context,
                        o_strategies=create_qneural_player,modele_strategie_O = net_context2)
    """print("Training qlearning O vs. random...")
    play_training_games_o(net_context=net_context,
                        x_strategies=[play_qneural_move])"""
    print("")
    #numba.cuda.profile_stop()

    with torch.no_grad():
        play_qneural_move = create_qneural_player(net_context)

        """print("Playing qneural vs random:")
        print("--------------------------")
        play_games(1000, play_qneural_move, play_random_move)
        print("")
        print("Playing qneural vs minimax random:")
        print("----------------------------------")
        play_games(1000, play_qneural_move, play_minimax_move_randomized)
        print("")
        print("Playing qneural vs minimax:")
        print("---------------------------")
        play_games(1000, play_qneural_move, play_minimax_move_not_randomized)
        print("")
        print(" ")
        print("Playing random vs qneural:")
        print("--------------------------")
        play_games(1000, play_random_move, play_qneural_move)
        print("")
        print("Playing minimax random vs qneural:")
        print("----------------------------------")
        play_games(1000, play_minimax_move_randomized, play_qneural_move)
        print("")

        print("Playing minimax vs qneural:")
        print("---------------------------")
        play_games(1000, play_minimax_move_not_randomized, play_qneural_move)
        print("  ")

        print("")
        print("Playing qneural vs minimax:")
        print("---------------------------")
        play_games(1000, play_qneural_move, play_minimax_move_not_randomized)
        print("")

        print("Playing qneural vs qneural:")
        print("---------------------------")
        play_games(1000, play_qneural_move, play_qneural_move)
        print("")"""

    

        """print("Playing qneural vs random:")
        print("--------------------------")
        play_games(100, play_human_move,play_qneural_move,)
        print("")"""
        

        """board = Board(np.array([1, -1, -1, 0, 1, 1, 0, 0, -1]))#, 0, 0, 0, 0, 0, 0, 0]))
        q_values = get_q_values(board, net_context.target_net)
        print(f"After training q_values = {q_values}")"""
