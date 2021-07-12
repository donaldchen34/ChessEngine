import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Concatenate
import chess
import chess.pgn
from pgn_parser import parser, pgn
import random
from tqdm import tqdm
from FeatureExtracter import Feature_Extractor

#363 - 90 Board Representation in bits "4.1 Feature Representation Giraffe"
#Missing 26 Features. Maybe convert some to more bit representation?
GLOBAL_FEATURES = 17
PIECE_CENTRIC_FEATURES = 192
SQUARE_CENTIC_FEATURES = 128
FEATURE_REPRESENTATION = GLOBAL_FEATURES + PIECE_CENTRIC_FEATURES + SQUARE_CENTIC_FEATURES
NUM = 3

POSSIBLE_MOVES = 4672 # 8x8x(8x7+8+9) -Change because no underpromotions
class TDLeaf:
    def __init__(self,alpha, gamma, num_episodes, moves_to_make, batch_size = 64):
        self.alpha = alpha #learning rate
        self.gamma = gamma #discount rate
        self.batch_size = batch_size

        self.num_episodes = num_episodes
        self.moves_to_make = moves_to_make

        self.feature_extractor = Feature_Extractor()

        self.model = self.createModel()

    #Returns string representaitons of the board in a list [position1,position2,...]
    #Reads from "CCRL-4040.[1259165].pgn/CCRL-4040.[1259165].pgn"
    #Wild inefficient
    # Multi process
    # Save work after done?
    # Optimize?
    def getTrainingData(self):
        """

        :return:
        """
        TRAINING_DATA_PATH = "CCRL-4040.[1259165].pgn/CCRL-4040.[1259165].pgn"
        training_pgn = open(TRAINING_DATA_PATH)

        chessPositions = []

        i = 0

        game_total = 1_259_165
        test_sample = 100
        #Chagnge total
        pbar = tqdm(total=test_sample)
        while True:
            if i >= test_sample:
                break

            try:
                game = chess.pgn.read_game(training_pgn)

            except:
                print("error?")
            else:
                chessPositions.extend(self.getChessPositions(game))
                i += 1
                pbar.update(1)


        pbar.close()
        print("Getting Data Done")

        return chessPositions

    def getChessPositions(self, game, random_positions = 5):
        """

        :param game: chess.Board
        :return: returns 5 random chess positions from a game
        """

        chess_positions = []

        #Not Optimal
        for i in range(random_positions):

            features = []

            moves = parser.parse(str(game.mainline_moves()), actions=pgn.Actions())
            game_len = len(moves.movetext)

            move_num = random.randint(0,(game_len-2)*2) # -2 so it is not the end of game, *2 because moves is pair of moves, 1.(white move, black move)
            board = chess.Board()
            j = 0
            #Probably a better way to do this
            for move in game.mainline_moves():
                if j >= move_num:
                    break

                board.push(move)
                j += 1

            features.extend(self.feature_extractor.getGlobalFeatures(board))
            features.extend(self.feature_extractor.getPieceCentricFeatures(board))
            features.extend(self.feature_extractor.getSquareCentricFeatures(board))

            chess_positions.append(features)

        return chess_positions

    def training(self):

        #Each training iteration:
        # Choose 256 positions from the training set
        # -> Apply 1 random move to each position
        # Self Play for 12 Turns
        # Record results for the 12 searches/moves
        # -> Update by adding changes over the 12 moves


        data = self.getTrainingData()

        for episode in self.num_episodes:

            for i in self.moves_to_make:

                pass
        pass

    #https://stackoverflow.com/questions/55233377/keras-sequential-model-with-multiple-inputs/55234203
    def createModel(self):

        global_features_input = Input(shape= GLOBAL_FEATURES,)
        x = Dense(GLOBAL_FEATURES * NUM, activation= 'relu')(global_features_input)
        piece_centric_input = Input(shape=PIECE_CENTRIC_FEATURES,)
        y = Dense(PIECE_CENTRIC_FEATURES * NUM, activation= 'relu')(piece_centric_input)
        square_centric_input = Input(shape= SQUARE_CENTIC_FEATURES,)
        z = Dense(SQUARE_CENTIC_FEATURES * NUM, activation= 'relu')(square_centric_input)
        merged = Concatenate(axis= 1)([x,y,z])
        t = Dense(FEATURE_REPRESENTATION,input_dim=3, activation= 'relu')(merged)
        output = Dense(POSSIBLE_MOVES, activation='tanh')(t)

        model = Model(inputs=[global_features_input,piece_centric_input,square_centric_input],
                      outputs=output)

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08, clipnorm=1.0))

        return model

def testTDLeaf():
    alpha = .2
    gamma = .7
    training_iterations = 5
    moves = 12



    test = TDLeaf(alpha=alpha, gamma=gamma,num_episodes=training_iterations,moves_to_make=moves)
    games = test.getTrainingData()
    print(len(games))
    print("WAT")

if __name__ == "__main__":
    testTDLeaf()
