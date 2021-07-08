#from tensorflow.keras.models import Model
#from tensorflow.keras.layers import Dense, Input, Concatenate
import chess
import chess.pgn
from pgn_parser import parser, pgn
from BoardRepresentation import convertBoardToList
import random
from tqdm import tqdm
from FeatureExtracter import Feature_Extractor

#363 - 90 Board Representation in bits "4.1 Feature Representation Giraffe"
#Missing 90 Features. Maybe convert some to more bit representation?
GLOBAL_FEATURES = 17
PIECE_CENTRIC_FEATURES = 160
SQUARE_CENTIC_FEATURES = 64
FEATURE_REPRESENTATION = GLOBAL_FEATURES + PIECE_CENTRIC_FEATURES + SQUARE_CENTIC_FEATURES
NUM = 3

POSSIBLE_MOVES = 4672 # 8x8x(8x7+8+9) -Change because no underpromotions
class TDLeaf:
    def __init__(self,alpha, gamma):
        self.alpha = alpha #learning rate
        self.gamma = gamma #discount rate
        self.feature_extractor = Feature_Extractor()


    #Returns string representaitons of the board in a list [position1,position2,...]
    #Reads from "CCRL-4040.[1259165].pgn/CCRL-4040.[1259165].pgn"
    #Wild inefficient
    # Multi process
    # Save work after done?
    # Optimize?
    def getTrainingData(self):

        TRAINING_DATA_PATH = "CCRL-4040.[1259165].pgn/CCRL-4040.[1259165].pgn"
        training_pgn = open(TRAINING_DATA_PATH)

        chessPositions = []

        i = 0

        game_total = 1_259_165
        #Chagnge total
        pbar = tqdm(total=1000)
        while True:
            if i >= 1000:
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
        print("Done")

        return chessPositions

    def getChessPositions(self, game):
        # 1. Randomly choose move that is not end of game
        # 2. Run board until that move
        # 3. Save board as training data
        # 4. Repeat 1-3 on same board 5 times
        # 5. Repeat 4 on all data
        # 6. Return Training_Data

        chess_positions = []
        RANDOM_POSITIONS = 5

        #Not Optimal
        for i in range(RANDOM_POSITIONS):

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

            chess_positions.append(convertBoardToList(board=board))

        return chess_positions

    def training(self,num_episodes, moves_to_make):

        for i in num_episodes:

            for j in moves_to_make:

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

        #model.compile()

        return model

def testTDLeaf():
    test = TDLeaf(alpha=0.2, gamma=0.7)
    games = test.getTrainingData()
    print(len(games))
    print("WAT")

if __name__ == "__main__":
    test_pgn = open('CCRL-4040.[1259165].pgn/CCRL-4040.[1259165].pgn')
    first_game = chess.pgn.read_game(test_pgn)
    print(str(first_game.mainline_moves()))
    #`https: // github.com / brettbates / pgn_parser
    game = parser.parse(str(first_game.mainline_moves()),actions=pgn.Actions())
    print(game.move(2))
    print(game.move(2).black.san)
    print(len(game.movetext))

    stop = 2
    i = 0
    board = first_game.board()
    for move in first_game.mainline_moves():
        if i > stop:
            break
        board.push(move)
        i+=1

    x = board.epd()
    print(x)
    print(type(x))
    print(x.count("p"))

    #testTDLeaf()
