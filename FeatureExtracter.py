from Environment import convertBoardToList
import chess

class Feature_Extractor():

    def getFeatures(self,board):
        features = []
        features.extend(self.getGlobalFeatures(board))
        features.extend(self.getPieceCentricFeatures(board))
        features.extend(self.getSquareCentricFeatures(board))

        return features

    #Gets Side to move, Castling Rights, Material Configuration(Details in Board.txt)
    def getGlobalFeatures(self, board):
        """
        Gets Side to move, Castling Rights, Material Configuration (Details in Board.txt)

        :param board: chess.Board()
        :return: [Side to Move, Castling Rights, Material Configuration]
        """
        features = []
        # Side to Move
        sideToMove = int(board.turn)

        # Castling Rights
        whiteLongCastle = int(board.has_queenside_castling_rights(1))
        whiteShortCastle = int(board.has_kingside_castling_rights(1))
        blackLongCastle = int(board.has_queenside_castling_rights(0))
        blackShortCastle = int(board.has_kingside_castling_rights(0))

        # Material Configuration
        WhiteKing = len(board.pieces(6, 1))
        WhiteQueen = len(board.pieces(5, 1))
        WhiteRook = len(board.pieces(4, 1))
        WhiteBishop = len(board.pieces(3, 1))
        WhiteKnight = len(board.pieces(2, 1))
        WhitePawn = len(board.pieces(1, 1))
        BlackKing = len(board.pieces(6, 0))
        BlackQueen = len(board.pieces(5, 0))
        BlackRook = len(board.pieces(4, 0))
        BlackBishop = len(board.pieces(3, 0))
        BlackKnight = len(board.pieces(2, 0))
        BlackPawn = len(board.pieces(1, 0))

        features.append(sideToMove)
        features.extend([whiteLongCastle, whiteShortCastle, blackLongCastle, blackShortCastle])
        features.extend([WhiteKing, WhiteQueen, WhiteRook, WhiteBishop, WhiteKnight, WhitePawn,
                         BlackKing, BlackQueen, BlackRook, BlackBishop, BlackKnight, BlackPawn])

        return features

    # Gets Existence, Position, Lowest Value Attacker/ Defender, Mobility of each piece
    def getPieceCentricFeatures(self, board):
        """
        #Gets Existence, Position, Lowest Value Attacker/ Defender, Mobility of each piece

        :param board: chess.Board
        :return: [Existence, Position, Lowest Value Attacker/ Defender/ Mobility for Piece 1, ...]
        """

        features = []
        pieces = ['p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8', 'n1', 'n2', 'b1', 'b2', 'r1', 'r2', 'q', 'k',
                  'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'N1', 'N2', 'B1', 'B2', 'R1', 'R2', 'Q', 'K']

        positions = self.getPiecePositions(board)
        for piece in pieces:
            # x & y go from 1-8, 0 means not there
            x = positions[piece]['x'] #int
            y = positions[piece]['y'] #int
            existence = 1 if x else 0
            lowest_value_attacker,lowest_value_defender = self.getLowestValueAttackerandDefender(board,x - 1,y - 1) if existence else (0, 0)
            mobility = self.getMobility(board,x - 1,y - 1) if existence else 0

            features.extend([existence, x, y, lowest_value_attacker, lowest_value_defender, mobility])

        return features

    def getSquareCentricFeatures(self, board):
        """
        Gets lowest attacker and defender for each square starting at A1 : pos = 0 , A2 : pos = 2, ...

        :param board: chess.Board
        """
        features = []

        #return x + (7-y) * 8

        for i in range(7,-1,-1):
            for j in range(8):
                lowest_attacker, lowest_defender = self.getLowestValueAttackerandDefender(board,j,i)
                features.extend([lowest_attacker,lowest_defender])

        return features

    def getPos(self,x, y):
        """
        :param x: x pos, int or char
        :param y: y pos, int or char
        :return: pos on board. Check docs chess.Squares -> chess.A1 = 0 ... chess.H8 = 63
        """
        if str(x).isalpha():
            x = ord(x) - 97

        if str(y).isalpha():
            y = ord(y) - 97

        return x + (7-y) * 8

    def getMobility(self, board, x, y):
        """
        :param board: chess.Board
        :param x: x pos (int)
        :param y: y pos (int)
        :return: get amount of spaces the piece on x,y can move
        """
        pos = self.getPos(x,y)
        piece = board.piece_type_at(pos)
        mobility = 0
        positions = [] #integers for positions on board


        if piece == chess.PAWN:
            #Pawns at starting position
            if board.turn and y == 6:
                positions.append(self.getPos(x,y-2))
            if not board.turn and y == 1:
                positions.append(self.getPos(x,y+2))

            # Check diagnols and piece above it
            new_y = y - 1 if board.turn else y + 1
            new_pos = self.getPos(x,new_y)
            positions.extend([new_pos-1,new_pos,new_pos+1])

        if piece == chess.KNIGHT:

            positions.extend([pos + 17, pos + 10,
                              pos - 6, pos - 15,
                              pos - 17, pos - 10,
                              pos + 6, pos + 15])

        if piece == chess.BISHOP:
            # Top left to bottom right \
            #Change to +-7
            bottom = min(x,y)
            new_x, new_y = x - bottom, y - bottom
            for i in range(8-max(new_x,new_y)):
                positions.append(self.getPos(new_x,new_y))
                new_x += 1
                new_y += 1

            #Bottom left to top right /
            #Might be prone to error change to +-9
            top = max(x,y)
            new_x, new_y = x + top, y - top
            for i in range(max(new_x, 8 - new_y)):
                positions.append(self.getPos(new_x,new_y))
                new_x -= 1
                new_y += 1

        if piece == chess.ROOK:
            #Vertical
            for i in range(8):
                positions.append(self.getPos(x,i))

            #Horizontal
            for i in range(8):
                positions.append(self.getPos(i,y))

        if piece == chess.QUEEN:
            #Vertical
            for i in range(8):
                positions.append(self.getPos(x,i))

            #Horizontal
            for i in range(8):
                positions.append(self.getPos(i,y))

            #Top Left to Bottom Right \
            bottom = min(x,y)
            new_x, new_y = x - bottom, y - bottom
            for i in range(8-max(new_x,new_y)):
                positions.append(self.getPos(new_x,new_y))
                new_x += 1
                new_y += 1

            #Bottom left to top right /
            top = max(x,y)
            new_x, new_y = x + top, y - top
            for i in range(max(new_x, 8 - new_y)):
                positions.append(self.getPos(new_x,new_y))
                new_x -= 1
                new_y += 1

        if piece == chess.KING:

            #Castling
            if board.turn:
                if board.has_queenside_castling_rights(1):
                    if y == 7:
                        positions.append(pos - 2)
                if board.has_kingside_castling_rights(1):
                    if y == 7:
                        positions.append(pos + 2)
            else:
                if board.has_queenside_castling_rights(0):
                    if y == 0:
                        positions.append(pos - 2)
                if board.has_kingside_castling_rights(0):
                    if y == 0:
                        positions.append(pos + 2)

            positions.extend([pos-1,pos+1]) #Left, Right
            #Top/Bottom
            new_y = y - 1
            new_pos = self.getPos(x,new_y)
            positions.extend([new_pos-1,new_pos,new_pos+1])
            new_y = y + 1
            new_pos = self.getPos(x, new_y)
            positions.extend([new_pos-1,new_pos,new_pos+1])

        for move in positions:
            try:
                if board.find_move(pos, move):
                    mobility += 1
            except:
                pass

        return mobility

    # Gets lowest value attacker and defender of position x,y on board
    def getLowestValueAttackerandDefender(self, board, x, y):
        """

        :param board: chess.Board
        :param x: x position of board (int)
        :param y: y position of board (int or char) -> char not tested
        :return: lowest_value_attacker, lower_value_defender
        """

        pos = self.getPos(x,y)

        attackers_board = board.attackers(not board.turn, pos)
        defenders_board = board.attackers(board.turn, pos)

        lowest_value_attacker = 100
        lowest_value_defender = 100

        if len(attackers_board):
            for attacker_pos, square in enumerate(attackers_board.tolist()):
                if square:
                    lowest_value_attacker = min(lowest_value_attacker,board.piece_type_at(attacker_pos))
        else:
            lowest_value_attacker = 0

        if len(defenders_board):
            for defender_pos, square in enumerate(defenders_board.tolist()):
                if square:
                    lowest_value_defender = min(lowest_value_defender, board.piece_type_at(defender_pos))
        else:
            lowest_value_defender = 0


        return lowest_value_attacker, lowest_value_defender

        # Does not properly assign pieces to their slots
        # Slots are filled by amount
        # White pawn on e should be assigned to slot P5 but if eaten the slot will be taken by any extra pawns

    def getPiecePositions(self, board):
        """

        :param board:
        :return:
        """

        # Actual Starting Positions
        """
        positions = {'k': {'x': 'e', 'y': '1'},
                     'q': {'x': 'd', 'y': '1'},
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
                     'K': {'x': 'e', 'y': '8'},
                     'Q': {'x': 'd', 'y': '8'},
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
        """
        positions = {'k':  {'x': 0, 'y': 0},
                     'q':  {'x': 0, 'y': 0},
                     'r1': {'x': 0, 'y': 0},
                     'r2': {'x': 0, 'y': 0},
                     'b1': {'x': 0, 'y': 0},
                     'b2': {'x': 0, 'y': 0},
                     'n1': {'x': 0, 'y': 0},
                     'n2': {'x': 0, 'y': 0},
                     'p1': {'x': 0, 'y': 0},
                     'p2': {'x': 0, 'y': 0},
                     'p3': {'x': 0, 'y': 0},
                     'p4': {'x': 0, 'y': 0},
                     'p5': {'x': 0, 'y': 0},
                     'p6': {'x': 0, 'y': 0},
                     'p7': {'x': 0, 'y': 0},
                     'p8': {'x': 0, 'y': 0},
                     'K':  {'x': 0, 'y': 0},
                     'Q':  {'x': 0, 'y': 0},
                     'R1': {'x': 0, 'y': 0},
                     'R2': {'x': 0, 'y': 0},
                     'B1': {'x': 0, 'y': 0},
                     'B2': {'x': 0, 'y': 0},
                     'N1': {'x': 0, 'y': 0},
                     'N2': {'x': 0, 'y': 0},
                     'P1': {'x': 0, 'y': 0},
                     'P2': {'x': 0, 'y': 0},
                     'P3': {'x': 0, 'y': 0},
                     'P4': {'x': 0, 'y': 0},
                     'P5': {'x': 0, 'y': 0},
                     'P6': {'x': 0, 'y': 0},
                     'P7': {'x': 0, 'y': 0},
                     'P8': {'x': 0, 'y': 0}
                     }

        board_list = convertBoardToList(board)

        for x, row in enumerate(board_list):
            for y, piece in enumerate(row):

                if piece != '.':
                    if piece == 'k' or piece == 'K' or piece == 'q' or piece == 'Q':
                        positions[piece]['x'] = x + 1 # +1 because 0 represents not there
                        positions[piece]['y'] = y + 1
                    else:
                        slot = 1

                        pieceslot = piece + str(slot)
                        while positions[pieceslot]['x'] != 0:
                            slot += 1
                            pieceslot = piece + str(slot)

                        positions[pieceslot]['x'] = x + 1
                        positions[pieceslot]['y'] = y + 1

        return positions

