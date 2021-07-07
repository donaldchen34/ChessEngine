#from tensorflow.keras.models import Model
#from tensorflow.keras.layers import Dense, Input, Concatenate
import chess
import chess.pgn
from pgn_parser import parser, pgn
from BoardRepresentation import convertBoardToList
import random
from tqdm import tqdm

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

    def featureExtraction(self,board):
        """
        :param board: string representation of a board
        """
        features = []
        features.extend(self.getGlobalFeatures())
        features.extend(self.getPieceCentricFeatures())
        features.extend(self.getSquareCentricFeatures())

        return features

    def getGlobalFeatures(self,board):
        """
        Gets Side to move, Castling Rights, Material Configuration (Details in Board.txt)

        :param board: chess.Board()
        """
        features = []
        #Side to Move
        sideToMove = board.turn

        #Castling Rights
        whiteLongCastle = board.has_queenside_castling_rights(1)
        whiteShortCastle = board.has_kingside_castling_rights(1)
        blackLongCastle = board.has_queenside_castling_rights(0)
        blackShortCastle = board.has_kingside_castling_rights(0)

        #Material Configuration
        WhiteKing = len(board.pieces(6,1))
        WhiteQueen = len(board.pieces(5,1))
        WhiteRook = len(board.pieces(4,1))
        WhiteBishop = len(board.pieces(3, 1))
        WhiteKnight = len(board.pieces(2,1))
        WhitePawn = len(board.pieces(1,1))
        BlackKing = len(board.pieces(6,0))
        BlackQueen = len(board.pieces(5,0))
        BlackRook = len(board.pieces(4,0))
        BlackBishop = len(board.pieces(3, 0))
        BlackKnight = len(board.pieces(2,0))
        BlackPawn = len(board.pieces(1,0))

        features.append(sideToMove)
        features.extend([whiteLongCastle,whiteShortCastle,blackLongCastle,blackShortCastle])
        features.extend([WhiteKing,WhiteQueen,WhiteRook,WhiteBishop,WhiteKnight,WhitePawn,
                        BlackKing,BlackQueen,BlackRook,BlackBishop,BlackKnight,BlackPawn])

        return features

    def getPieceCentricFeatures(self, board):
        """
        Gets Existence, Position, Lowest Value Attacker/ Defender, Mobility of each piece

        :param board: string representation of a board
        """

        features = []
        pieces = ['p1','p2','p3','p4','p5','p6','p7','p8','n1','n2','b1','b2''r1','r2','q','k',
                  'P1','P2','P3','P4','P5','P6','P7','P8','N1','N2','B1','B2','R1','R2','Q','K']

        positions = self.getPiecePositions(board)
        for piece in pieces:

            x = positions[piece]['x']
            y = positions[piece]['y']
            existence = 1 if x else 0
            lowest_value_attacker = self.getLowestValueAttacker(board,x,y) if x else 0
            lowest_value_defender = 0
            mobility = 0

            features.extend([existence,x,y,lowest_value_attacker,lowest_value_defender,mobility])

        return features

    def getLowestValueAttacker(self,board, x, y):
        """
        squares1 = self.getBoard().is_attacked_by(not self.board.turn, pos)
        squares2 = self.getBoard().attacks(pos)
        squares3 = self.getBoard().attackers(not self.board.turn, pos)
        print('attacked by:')
        print(squares1)
        print('attacks:')
        print(squares2)
        print(len(squares2))
        print('attackers:')
        print(squares3)
        print(len(squares3))
        """

        PIECE_TYPES = {}

        num_y = ord(y) - 97
        pos = x * 8 + num_y
        attackers = board.attackers(board.turn, pos)
        if len(attackers):
            for attacker_pos,square in enumerate(attackers.tolist()):
                if square:
                    attackers.piece_type_at(attacker_pos)

        else:
            return 0


    #Does not properly assign pieces to their slots
    #Slots are filled by amount
    #White pawn on e should be assigned to slot P5 but if eaten the slot will be taken by any extra pawns
    def getPiecePositions(self,board):

        #Actual Starting Positions
        positions = {'k':  {'x': 'e', 'y': '1'},
                     'q':  {'x': 'd', 'y': '1'},
                     'r1': {'x': 'a', 'y': '1'},
                     'r2': {'x': 'h', 'y': '1'},
                     'b1': {'x': 'c', 'y': '1'},
                     'b2': {'x': 'f', 'y': '1'},
                     'n1': {'x': 'b', 'y': '1'},
                     'n2': {'x': 'g', 'y': '1'},
                     'p1': {'x': 'a', 'y': '2'},
                     'p2': {'x': 'b', 'y': '2'},
                     'p3': {'x': 'c', 'y': '2'},
                     'p4': {'x': 'd', 'y': '2'},
                     'p5': {'x': 'e', 'y': '2'},
                     'p6': {'x': 'f', 'y': '2'},
                     'p7': {'x': 'g', 'y': '2'},
                     'p8': {'x': 'h', 'y': '2'},
                     'K':  {'x': 'e', 'y': '8'},
                     'Q':  {'x': 'd', 'y': '8'},
                     'R1': {'x': 'a', 'y': '8'},
                     'R2': {'x': 'h', 'y': '8'},
                     'B1': {'x': 'c', 'y': '8'},
                     'B2': {'x': 'f', 'y': '8'},
                     'N1': {'x': 'b', 'y': '8'},
                     'N2': {'x': 'g', 'y': '8'},
                     'P1': {'x': 'a', 'y': '7'},
                     'P2': {'x': 'b', 'y': '7'},
                     'P3': {'x': 'c', 'y': '7'},
                     'P4': {'x': 'd', 'y': '7'},
                     'P5': {'x': 'e', 'y': '7'},
                     'P6': {'x': 'f', 'y': '7'},
                     'P7': {'x': 'g', 'y': '7'},
                     'P8': {'x': 'h', 'y': '7'}
                     }

        positions = {'k':  {'x': '0', 'y': '0'},
                     'q':  {'x': '0', 'y': '0'},
                     'r1': {'x': '0', 'y': '0'},
                     'r2': {'x': '0', 'y': '0'},
                     'b1': {'x': '0', 'y': '0'},
                     'b2': {'x': '0', 'y': '0'},
                     'n1': {'x': '0', 'y': '0'},
                     'n2': {'x': '0', 'y': '0'},
                     'p1': {'x': '0', 'y': '0'},
                     'p2': {'x': '0', 'y': '0'},
                     'p3': {'x': '0', 'y': '0'},
                     'p4': {'x': '0', 'y': '0'},
                     'p5': {'x': '0', 'y': '0'},
                     'p6': {'x': '0', 'y': '0'},
                     'p7': {'x': '0', 'y': '0'},
                     'p8': {'x': '0', 'y': '0'},
                     'K':  {'x': '0', 'y': '0'},
                     'Q':  {'x': '0', 'y': '0'},
                     'R1': {'x': '0', 'y': '0'},
                     'R2': {'x': '0', 'y': '0'},
                     'B1': {'x': '0', 'y': '0'},
                     'B2': {'x': '0', 'y': '0'},
                     'N1': {'x': '0', 'y': '0'},
                     'N2': {'x': '0', 'y': '0'},
                     'P1': {'x': '0', 'y': '0'},
                     'P2': {'x': '0', 'y': '0'},
                     'P3': {'x': '0', 'y': '0'},
                     'P4': {'x': '0', 'y': '0'},
                     'P5': {'x': '0', 'y': '0'},
                     'P6': {'x': '0', 'y': '0'},
                     'P7': {'x': '0', 'y': '0'},
                     'P8': {'x': '0', 'y': '0'}
                    }

        board_list = convertBoardToList(board)

        for x,row in enumerate(board_list):
            for y,piece in row:
                if piece != '.':
                    if piece == 'k' or piece == 'K' or piece == 'q' or piece == 'Q':
                        positions[piece]['x'] = x
                        positions[piece]['y'] = chr(y+97)
                    else:
                        slot = 1
                        pieceslot = piece + slot
                        while positions[pieceslot][x] == 0:
                            slot += 1
                            pieceslot = piece + slot

                        positions[pieceslot]['x'] = x
                        positions[pieceslot]['y'] = chr(y+97)

        return positions

    def getSquareCentricFeatures(self, board):
        """
        :param board: string representation of a board
        """
        pass

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
