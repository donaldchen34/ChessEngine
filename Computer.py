import random
from BoardRepresentation import Evaluator
import copy
from TDLeaf import TDLeaf
import numpy as np
from FeatureExtracter import FeatureExtractor


class Computer:
    def __init__(self, board=None, depth=1, algo='random'):
        self.algo = algo
        self.board = board  # chess.board
        self.depth = depth  # used for minimax

        self.evaluator = Evaluator()

    def set_board(self, board):
        self.board = board

    def random(self, board=None):
        new_board = board if board else self.board
        moves = [move for move in new_board.legal_moves]
        comp_move = random.randint(0, len(moves) - 1)
        new_board.push(moves[comp_move])

    # Returns move, evaluation
    def minimax(self, board, depth, maximizing_player):
        return self.minimax_helper(board=board, depth=depth, alpha=-100000, beta=100000, maximizing_player=maximizing_player)

    # Todo
    # Evaluator is too basic
    # TurnCount is incorrect
    # Multiprocess
    # Add support for same value best move(random from list)
    def minimax_helper(self, board, depth, alpha, beta, maximizing_player):
        if depth == 0 or board.is_game_over():
            turn = board.fullmove_number * 2 + 1 if board.turn else board.fullmove_number * 2  # Line used in TDLeaf too
            return None, self.evaluator.getEval(board=board, turn_count=turn, turn=not board.turn)

        moves = [move for move in board.legal_moves]
        best_move = moves[0]

        if maximizing_player:
            value = -10000
            for move in moves:
                board_copy = copy.deepcopy(board)
                board_copy.push(move)
                current_eval = self.minimax_helper(board_copy, depth-1, alpha, beta, False)[1]
                if current_eval > value:
                    value = current_eval
                    best_move = move
                alpha = max(alpha, current_eval)
                if beta <= alpha:
                    break

            return best_move, value
        else:
            value = 10000
            for move in moves:
                board_copy = copy.deepcopy(board)
                board_copy.push(move)
                current_eval = self.minimax_helper(board_copy, depth - 1, alpha, beta, True)[1]
                if current_eval < value:
                    value = current_eval
                    best_move = move

                beta = min(beta, current_eval)
                if beta <= alpha:
                    break

            return best_move, value

    def neural_network(self, board=None):
        # Load trained model
        # -> Use model to predict evaluation on each possible move
        #   -> Use best move

        model = TDLeaf.create_model()
        model.load_weights('weights.h5')

        feature_extractor = FeatureExtractor()

        board = board if board else self.board
        moves = [move for move in board.legal_moves]

        best_move = moves[0]
        best_val = -1  # the output neuron has a tanh function
        for move in moves:
            board_copy = copy.deepcopy(board)
            board_copy.push(move)

            global_features = np.array([feature_extractor.get_global_features(board_copy)])
            piece_centric_features = np.array([feature_extractor.get_piece_centric_features(board_copy)])
            square_centric_features = np.array([feature_extractor.get_square_centric_features(board_copy)])
            x = model([global_features, piece_centric_features, square_centric_features])
            print(x)
            x = x.numpy()[0][0]  # x is stored as a [[evaluation]]
            print(x , board_copy.turn)

            x = -x if board_copy.turn else x  # Inverse if black turn
            print(x)
            print('------------')
            if x > best_val:
                best_val = x
                best_move = move

        board.push(best_move)



    def make_move(self):
        if self.algo == 'random':
            self.random()
        if self.algo == "minimax":
            move, x = self.minimax(self.board, self.depth, self.board.turn)
            self.board.push(move)
        if self.algo == "dqn":
            self.neural_network()
