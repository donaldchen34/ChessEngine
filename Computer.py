from QLearning import DQN
import random
class Computer:
    def __init__(self, board, algo = 'random'):
        self.algo = algo
        self.board = board

        pass

    def random(self):
        moves = [move for move in self.board.legal_moves]
        comp_move = random.randint(0, len(moves) - 1)
        self.board.push(moves[comp_move])

        pass

    def minmax(self):
        pass

    def neuralnetwork(self):
        DQN()
        pass

    def makeMove(self):
        if self.algo == 'random':
            self.random()
        if self.algo == "minmax":
            self.minmax()
        if self.algo == "dqn":
            self.neuralnetwork()