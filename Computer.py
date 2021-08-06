import random
from BoardRepresentation import Evaluator
import copy
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
            return None, self.evaluator.get_eval(board=board, turn_count=turn, turn=not board.turn)

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

        model = self.create_model()
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
            x = x.numpy()[0][0]  # x is stored as a [[evaluation]]

            # x = x if board_copy.turn else -x  # Inverse if black turn

            if x > best_val:
                best_val = x
                best_move = move

        board.push(best_move)

    # Copied from TDLeaf
    def create_model(self):
        import tensorflow as tf
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import Dense, Input, concatenate

        # 363 - 90 Board Representation in bits "4.1 Feature Representation Giraffe"
        # Missing 26 Features
        GLOBAL_FEATURES = 17
        PIECE_CENTRIC_FEATURES = 192
        SQUARE_CENTIC_FEATURES = 128
        FEATURE_REPRESENTATION = GLOBAL_FEATURES + PIECE_CENTRIC_FEATURES + SQUARE_CENTIC_FEATURES

        # Following the rules:
        # The number of hidden neurons should be:
        # 1. Between the size of the input layer and the size of the output layer
        # 2. 2/3 the size of the input layer plus the size of the output layer
        # 3. less than twice the size of the input layer
        # -> The two hidden layers are broken down into thirds of the input size

        # +1 in Dense layer for x, y, z to represent the bias
        global_features_input = Input(shape=(GLOBAL_FEATURES,))
        x = Dense(int(GLOBAL_FEATURES / 3), activation='relu', use_bias=True)(global_features_input)
        x = Model(inputs=global_features_input, outputs=x)

        piece_centric_input = Input(shape=(PIECE_CENTRIC_FEATURES,))
        y = Dense(int(PIECE_CENTRIC_FEATURES / 3), activation='relu', use_bias=True)(piece_centric_input)
        y = Model(inputs=piece_centric_input, outputs=y)

        square_centric_input = Input(shape=(SQUARE_CENTIC_FEATURES,))
        z = Dense(int(SQUARE_CENTIC_FEATURES / 3), activation='relu', use_bias=True)(square_centric_input)
        z = Model(inputs=square_centric_input, outputs=z)

        merged = concatenate([x.output, y.output, z.output])
        t = Dense(int(FEATURE_REPRESENTATION / 3), input_dim=4, activation='relu', use_bias=True)(merged)
        output = Dense(1, activation='tanh')(t)

        model = Model(inputs=[x.input, y.input, z.input],
                      outputs=output)

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08, clipnorm=1.0))

        return model

    def make_move(self):
        if self.algo == 'random':
            self.random()
        if self.algo == "minimax":
            move, x = self.minimax(self.board, self.depth, self.board.turn)
            self.board.push(move)
        if self.algo == "dqn":
            self.neural_network()
