import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, concatenate
import chess
import chess.pgn
from pgn_parser import parser, pgn
import random
from tqdm import tqdm
from FeatureExtracter import FeatureExtractor
import numpy as np
from BoardRepresentation import Evaluator
import copy
from Computer import Computer

# 363 - 90 Board Representation in bits "4.1 Feature Representation Giraffe"
# Missing 26 Features
GLOBAL_FEATURES = 17
PIECE_CENTRIC_FEATURES = 192
SQUARE_CENTIC_FEATURES = 128
FEATURE_REPRESENTATION = GLOBAL_FEATURES + PIECE_CENTRIC_FEATURES + SQUARE_CENTIC_FEATURES


class TDLeaf:
    def __init__(self, alpha, gamma, training_iterations, moves_to_make, batch_size=256):
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # discount rate

        self.batch_size = batch_size

        self.training_iterations = training_iterations
        self.moves_to_make = moves_to_make

        self.feature_extractor = FeatureExtractor()

        self.model = self.create_model()
        self.basic_eval = Evaluator()

    def training(self):

        # Each training iteration:
        # 1. Choose 256 positions from the training set
        # 2. -> Apply 1 random move to each position
        # 3. Self Play for 12 Turns
        # 4. Record results for the 12 searches/moves
        # 5. ->  Update by adding changes over the 12 moves

        data = self.get_training_data()
        data_len = len(data)

        computer = Computer(algo='minimax',depth=1)

        print("Training Iterations")
        for episode in tqdm(range(self.training_iterations)):
            print("Training batch")

            total_errors = []
            states = []

            for i in tqdm(range(self.batch_size)):  # 1 #Change to randsample and read for each game
                game_num = random.randint(0, data_len - 1)
                game = data[game_num]  # 2 - Random positions already applied

                initial_state = copy.deepcopy(game)
                rewards = []
                states.append(initial_state)

                for j in range(self.moves_to_make):  # 3 - Self play for 12 Turns
                    if game.is_game_over():
                        break

                    # Not sure if a turn is considered one move or two moves
                    # Reward may be skewed. Need to scale it down
                    action, reward_1 = computer.minimax(game, 1, game.turn)

                    # turn is not accurate
                    turn = game.fullmove_number * 2 + 1 if game.turn else game.fullmove_number * 2
                    reward_0 = self.basic_eval.get_eval(board=game, turn_count=turn, turn=game.turn)
                    game.push(action)

                    reward = reward_1 - reward_0

                    """
                    # Add a bonus or penalty for winning/ losing
                    if game.is_game_over():
                        # Rewards may be too large
                        outcome = game.outcome()
                        if game.is_stalemate():
                            reward = 50000 if game.turn else -50000
                        elif outcome.winner == turn:
                            reward = 100000
                        else:
                            reward = -100000
                    """

                    # reward = -reward if game.turn else reward
                    rewards.append(reward)

                # Get total error
                total_error = 0
                for x, reward in enumerate(rewards):
                    total_error += reward * self.gamma ** x
                total_errors.append(total_error)

            # 5 - Update weights

            # Update Gradients
            for state, error in zip(states,total_errors):
                global_features = np.array([self.feature_extractor.get_global_features(state)])
                piece_centric_features = np.array([self.feature_extractor.get_piece_centric_features(state)])
                square_centric_features = np.array([self.feature_extractor.get_square_centric_features(state)])

                with tf.GradientTape() as tape:
                    # Use L1 Loss Function
                    loss = self.model([global_features, piece_centric_features, square_centric_features])
                    loss_func = tf.reduce_sum(abs(loss - error))

                gradient = tape.gradient(loss_func, self.model.trainable_weights)

                opt = tf.keras.optimizers.Adadelta(learning_rate=0.001)
                opt.apply_gradients(zip(gradient,self.model.trainable_weights))

        # Save model
        self.model.save_weights('weights.h5')

    # Returns string representations of the board in a list [position1,position2,...]
    # Reads from "CCRL-4040.[1259165].pgn/CCRL-4040.[1259165].pgn"
    # Wild inefficient
    # -> Optimize?
    # Multi process?
    # Save work after done?
    def get_training_data(self):
        """
        :return: Returns board positions with random moves applied to them from the pgn file
        """
        TRAINING_DATA_PATH = "CCRL-4040.[1259165].pgn/CCRL-4040.[1259165].pgn"
        training_pgn = open(TRAINING_DATA_PATH)

        chess_positions = []

        i = 0

        game_total = 1_259_165  # amount of games in the training_pgn
        test_sample = 500
        # Change total
        pbar = tqdm(total=test_sample)
        print("Getting Training Data")
        while True:
            if i >= test_sample:
                break
            game = chess.pgn.read_game(training_pgn)
            chess_positions.extend(self.get_chess_positions(game))
            i += 1
            pbar.update(1)

        pbar.close()

        print("Getting Training Data Done")

        return chess_positions

    # Gets 5 random positions from a game and applies a random move to it
    # Used to increase training_sample_size
    @staticmethod
    def get_chess_positions(game, random_positions=5):
        """
        :param random_positions: amount of positions to get from a chess.board game
        :param game: chess.Board
        :return: returns random_position amount of random chess.Board positions from a game
        """
        chess_positions = []

        # Not Optimal
        for i in range(random_positions):

            moves = parser.parse(str(game.mainline_moves()), actions=pgn.Actions())
            game_len = len(moves.movetext)

            # -2 so it is not the end of game, *2 because moves is pair of moves, 1.(white move, black move)
            move_num = random.randint(0, (game_len-2)*2)
            board = chess.Board()
            j = 0
            # Probably a better way to do this -> chess library might have someting
            for move in game.mainline_moves():
                if j >= move_num:
                    break

                board.push(move)
                j += 1

            chess_positions.append(board)

        return chess_positions

    # Todo
    # Add biases
    @staticmethod
    def create_model():

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


def test_TDLeaf():
    alpha = 1
    gamma = .7
    training_iterations = 25
    moves = 6

    test = TDLeaf(alpha=alpha, gamma=gamma, training_iterations=training_iterations, moves_to_make=moves)
    test.training()


if __name__ == "__main__":
    test_TDLeaf()


