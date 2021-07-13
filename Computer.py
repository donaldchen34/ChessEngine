import random
from BoardRepresentation import Evaluator, convertBoardToList
import copy

class Computer:
    def __init__(self, board = None, depth = 1,algo = 'random'):
        self.algo = algo
        self.board = board
        self.depth = depth
        self.evaluator = Evaluator()

        pass

    def random(self,board = None):
        new_board = board if board else self.board
        moves = [move for move in new_board.legal_moves]
        comp_move = random.randint(0, len(moves) - 1)
        new_board.push(moves[comp_move])

    #Evaluator is too basic
    #TurnCount is incorrect
    def minimax(self, board, depth, alpha, beta, maximizingPlayer):

        if depth == 0 or board.is_game_over():
            turn = board.fullmove_number * 2 + 1 if board.turn else board.fullmove_number * 2
            return None, self.evaluator.getEval(board=board,turn_count=turn,turn= not board.turn)

        moves = [move for move in board.legal_moves]
        best_move = moves[0]

        if maximizingPlayer:
            value = -10000
            for move in moves:
                board_copy = copy.deepcopy(board)
                board_copy.push(move)
                current_eval = self.minimax(board_copy,depth-1, alpha, beta, False)[1]
                if current_eval > value:
                    value = current_eval
                    best_move = move
                alpha = max(alpha,current_eval)
                if beta <= alpha:
                    break

            return best_move, value
        else:
            value = 10000
            for move in moves:
                board_copy = copy.deepcopy(board)
                board_copy.push(move)
                current_eval = self.minimax(board_copy, depth - 1, alpha, beta, True)[1]
                if current_eval < value:
                    value = current_eval
                    best_move = move

                beta = min(beta,current_eval)
                if beta <= alpha:
                    break

            return best_move, value

    def neuralnetwork(self):
        pass

    def makeMove(self):
        if self.algo == 'random':
            self.random()
        if self.algo == "minimax":
            move = self.minimax(self.board, self.depth, -100000, 100000, True)[0]
            self.board.push(move)

        if self.algo == "dqn":
            self.neuralnetwork()