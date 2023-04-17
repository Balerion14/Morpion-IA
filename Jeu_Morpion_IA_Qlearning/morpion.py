import numpy as np
import pickle

BOARD_ROWS = 5
BOARD_COLS = 5


class State:
    def __init__(self, p1, p2):
        self.board = np.zeros((BOARD_ROWS, BOARD_COLS))
        self.p1 = p1
        self.p2 = p2
        self.isEnd = False
        self.boardHash = None
        # init p1 plays first
        self.playerSymbol = 1

    # get unique hash of current board state
    def getHash(self):
        self.boardHash = str(self.board.reshape(BOARD_COLS * BOARD_ROWS))
        return self.boardHash

    def winner(self):
        joueur1 = 1
        joueur2 = -1
        pions = 4
        plateau = self.board

        # vérifier la victoire pour chaque joueur
        for joueur in [joueur1, joueur2]:
            # parcourir les lignes
            for i in range(plateau.shape[0]):
                for j in range(plateau.shape[1]):
                    if plateau[i][j] == joueur:
                        nb_pions = 1  # on commence à 1 car on a déjà un pion
                        # on vérifie à droite
                        k = 1
                        while j + k < plateau.shape[1] and plateau[i][j + k] == joueur:
                            nb_pions += 1
                            k += 1
                        # on vérifie à gauche
                        k = 1
                        while j - k >= 0 and plateau[i][j - k] ==joueur:
                            nb_pions += 1
                            k += 1
                        # si le joueur a aligné pions pions, il a gagné
                        if nb_pions >= pions:
                            self.isEnd = True
                            return joueur

            # parcourir les colonnes
            for i in range(plateau.shape[0]):
                for j in range(plateau.shape[1]):
                    if plateau[i][j] == joueur:
                        nb_pions = 1  # on commence à 1 car on a déjà un pion
                        # on vérifie en bas
                        k = 1
                        while i + k < plateau.shape[0] and plateau[i + k][j] == joueur:
                            nb_pions += 1
                            k += 1
                        # on vérifie en haut
                        k = 1
                        while i - k >= 0 and plateau[i - k][j] == joueur:
                            nb_pions += 1
                            k += 1
                        # si le joueur a aligné pions pions, il a gagné
                        if nb_pions >= pions:
                            self.isEnd = True
                            return joueur

            # parcourir les diagonales
            for i in range(plateau.shape[0]):
                for j in range(plateau.shape[1]):
                    if plateau[i][j] == joueur:
                        nb_pions = 1  # on commence à 1 car on a déjà un pion
                        # on vérifie en bas à droite
                        k = 1
                        while i + k < plateau.shape[0] and j + k < plateau.shape[1] and plateau[i + k][j + k] == joueur:
                            nb_pions += 1
                            k += 1
                        # on vérifie en haut à gauche
                        k = 1
                        while i - k >= 0 and j - k >= 0 and plateau[i - k][j - k] == joueur:
                            nb_pions += 1
                            k += 1
                        # si le joueur a aligné pions pions, il a gagné
                        if nb_pions >= pions:
                            self.isEnd = True
                            return joueur

            # parcourir les diagonales inverses
            for i in range(plateau.shape[0]):
                for j in range(plateau.shape[1]):
                    if plateau[i][j] == joueur:
                        nb_pions = 1  # on commence à 1 car on a déjà un pion
                        # on vérifie en bas à gauche
                        k = 1
                        while i + k < plateau.shape[0] and j - k >= 0 and plateau[i + k][j - k] == joueur:
                            nb_pions += 1
                            k += 1
                        # on vérifie en haut à droite
                        k = 1
                        while i - k >= 0 and j + k < plateau.shape[1] and plateau[i - k][j + k] == joueur:
                            nb_pions += 1
                            k += 1
                        # si le joueur a aligné pions pions, il a gagné
                        if nb_pions >= pions:
                            self.isEnd = True
                            return joueur

        # tie
        # no available positions
        if len(self.availablePositions()) == 0:
            self.isEnd = True
            return 0
        # not end
        self.isEnd = False
        return None

    def availablePositions(self):
        positions = []
        for i in range(BOARD_ROWS):
            for j in range(BOARD_COLS):
                if self.board[i, j] == 0:
                    positions.append((i, j))  # need to be tuple
        return positions

    def updateState(self, position):
        self.board[position] = self.playerSymbol
        # switch to another player
        self.playerSymbol = -1 if self.playerSymbol == 1 else 1

    # only when game ends
    def giveReward(self):
        result = self.winner()
        # backpropagate reward
        if result == 1:
            self.p1.feedReward(1)
            self.p2.feedReward(0)
        elif result == -1:
            self.p1.feedReward(0)
            self.p2.feedReward(1)
        else:
            self.p1.feedReward(0.1)
            self.p2.feedReward(0.5)

    # board reset
    def reset(self):
        self.board = np.zeros((BOARD_ROWS, BOARD_COLS))
        self.boardHash = None
        self.isEnd = False
        self.playerSymbol = 1

    def play(self, rounds=100):
        for i in range(rounds):
            if i % 1000 == 0:
                print("Rounds {}".format(i))
            if (i+1) % (rounds / 10) == 0:
                self.p1.exp_rate = max(0, self.p1.exp_rate - 0.1)
                self.p2.exp_rate = max(0, self.p2.exp_rate - 0.1)
                print(f"{i+1}/{rounds} games, using epsilon={self.p1.exp_rate}...")
            while not self.isEnd:
                # Player 1
                positions = self.availablePositions()
                p1_action = self.p1.chooseAction(positions, self.board, self.playerSymbol)
                # take action and upate board state
                self.updateState(p1_action)
                board_hash = self.getHash()
                self.p1.addState(board_hash)
                # check board status if it is end

                win = self.winner()
                if win is not None:
                    # self.showBoard()
                    # ended with p1 either win or draw
                    self.giveReward()
                    self.p1.reset()
                    self.p2.reset()
                    self.reset()
                    break

                else:
                    # Player 2
                    positions = self.availablePositions()
                    p2_action = self.p2.chooseAction(positions, self.board, self.playerSymbol)
                    self.updateState(p2_action)
                    board_hash = self.getHash()
                    self.p2.addState(board_hash)

                    win = self.winner()
                    if win is not None:
                        # self.showBoard()
                        # ended with p2 either win or draw
                        self.giveReward()
                        self.p1.reset()
                        self.p2.reset()
                        self.reset()
                        break

    # play with human
    def play2(self):
        while not self.isEnd:
            # Player 1
            positions = self.availablePositions()
            p1_action = self.p1.chooseAction(positions, self.board, self.playerSymbol)
            # take action and upate board state
            self.updateState(p1_action)
            self.showBoard()
            # check board status if it is end
            win = self.winner()
            if win is not None:
                if win == 1:
                    print(self.p1.name, "wins!")
                else:
                    print("tie!")
                self.reset()
                break

            else:
                # Player 2
                positions = self.availablePositions()
                p2_action = self.p2.chooseAction(positions)

                self.updateState(p2_action)
                self.showBoard()
                win = self.winner()
                if win is not None:
                    if win == -1:
                        print(self.p2.name, "wins!")
                    else:
                        print("tie!")
                    self.reset()
                    break

    def showBoard(self):
        # p1: x  p2: o
        for i in range(0, BOARD_ROWS):
            print('-------------')
            out = '| '
            for j in range(0, BOARD_COLS):
                if self.board[i, j] == 1:
                    token = 'x'
                if self.board[i, j] == -1:
                    token = 'o'
                if self.board[i, j] == 0:
                    token = ' '
                out += token + ' | '
            print(out)
        print('-------------')


class Player:
    def __init__(self, name, exp_rate=0.9, decay_rate=0.99995):
        self.name = name
        self.states = []  # record all positions taken
        self.lr = 0.2
        self.exp_rate = exp_rate
        self.decay_rate = decay_rate
        self.decay_gamma = 0.9
        self.states_value = {}  # state -> value

    def getHash(self, board):
        boardHash = str(board.reshape(BOARD_COLS * BOARD_ROWS))
        return boardHash

    def chooseAction(self, positions, current_board, symbol):
        if np.random.uniform(0, 1) <= self.exp_rate:
            # take random action
            idx = np.random.choice(len(positions))
            action = positions[idx]
        else:
            value_max = -999
            for p in positions:
                next_board = current_board.copy()
                next_board[p] = symbol
                next_boardHash = self.getHash(next_board)
                value = 0 if self.states_value.get(next_boardHash) is None else self.states_value.get(next_boardHash)
                # print("value", value)
                if value >= value_max:
                    value_max = value
                    action = p
        # print("{} takes action {}".format(self.name, action))
        return action

    # append a hash state
    def addState(self, state):
        self.states.append(state)

    # at the end of game, backpropagate and update states value
    def feedReward(self, reward):
        for st in reversed(self.states):
            if self.states_value.get(st) is None:
                self.states_value[st] = 0
            self.states_value[st] += self.lr * (self.decay_gamma * reward - self.states_value[st])
            reward = self.states_value[st]

    def reset(self):
        self.states = []

    def savePolicy(self):
        fw = open('policy_' + str(self.name), 'wb')
        pickle.dump(self.states_value, fw)
        fw.close()

    def loadPolicy(self, file):
        fr = open(file, 'rb')
        self.states_value = pickle.load(fr)
        fr.close()


class HumanPlayer:
    def __init__(self, name):
        self.name = name

    def chooseAction(self, positions):
        while True:
            row = int(input("Input your action row:"))
            col = int(input("Input your action col:"))
            action = (row, col)
            if action in positions:
                return action

    # append a hash state
    def addState(self, state):
        pass

    # at the end of game, backpropagate and update states value
    def feedReward(self, reward):
        pass

    def reset(self):
        pass


if __name__ == "__main__":
    # training
    p1 = Player("p1")
    p2 = Player("p2")

    st = State(p1, p2)
    print("training...")
    st.play(500000)#300000

    # Enregistrer les politiques des joueurs
    p1.savePolicy()
    p2.savePolicy()

    # play with human
    p1 = Player("computer", exp_rate=0)
    p1.loadPolicy("policy_p1")

    p2 = HumanPlayer("human")

    st = State(p1, p2)
    st.play2()