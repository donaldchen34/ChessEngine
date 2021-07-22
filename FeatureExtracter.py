from BoardRepresentation import convertBoardToList
import chess


class FeatureExtractor:

    def get_features(self, board):  # 337 Features
        features = []
        features.extend(self.get_global_features(board))
        features.extend(self.get_piece_centric_features(board))
        features.extend(self.get_square_centric_features(board))

        return features

    # Gets Side to move, Castling Rights, Material Configuration(Details in Board.txt) - 17 Features
    def get_global_features(self, board):
        """
        Gets Side to move, Castling Rights, Material Configuration (Details in Board.txt)

        :param board: chess.Board()
        :return: [Side to Move, Castling Rights, Material Configuration]
        """
        features = []
        # Side to Move
        side_to_move = int(board.turn)

        # Castling Rights
        white_long_castle = int(board.has_queenside_castling_rights(1))
        white_short_castle = int(board.has_kingside_castling_rights(1))
        black_long_castle = int(board.has_queenside_castling_rights(0))
        black_short_castle = int(board.has_kingside_castling_rights(0))

        # Material Configuration
        white_king = len(board.pieces(6, 1))
        white_queen = len(board.pieces(5, 1))
        white_rook = len(board.pieces(4, 1))
        white_bishop = len(board.pieces(3, 1))
        white_knight = len(board.pieces(2, 1))
        white_pawn = len(board.pieces(1, 1))
        black_king = len(board.pieces(6, 0))
        black_queen = len(board.pieces(5, 0))
        black_rook = len(board.pieces(4, 0))
        black_bishop = len(board.pieces(3, 0))
        black_knight = len(board.pieces(2, 0))
        black_pawn = len(board.pieces(1, 0))

        features.append(side_to_move)
        features.extend([white_long_castle, white_short_castle, black_long_castle, black_short_castle])
        features.extend([white_king, white_queen, white_rook, white_bishop, white_knight, white_pawn,
                         black_king, black_queen, black_rook, black_bishop, black_knight, black_pawn])

        return features

    # Gets Existence, Position, Lowest Value Attacker/ Defender, Mobility of each piece - 192 Features
    def get_piece_centric_features(self, board):
        """
        #Gets Existence, Position, Lowest Value Attacker/ Defender, Mobility of each piece

        :param board: chess.Board
        :return: [Existence, Position, Lowest Value Attacker/ Defender/ Mobility for Piece 1, ...]
        """

        features = []
        pieces = ['p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8', 'n1', 'n2', 'b1', 'b2', 'r1', 'r2', 'q', 'k',
                  'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'N1', 'N2', 'B1', 'B2', 'R1', 'R2', 'Q', 'K']

        positions = self.get_piece_positions(board)
        for piece in pieces:
            # x & y go from 1-8, 0 means not there
            x = positions[piece]['x']  # int
            y = positions[piece]['y']  # int
            existence = 1 if x else 0
            lowest_value_attacker, lowest_value_defender = \
                self.get_lowest_value_attacker_and_defender(board, x - 1, y - 1) if existence else (0, 0)
            mobility = self.get_mobility(board, x - 1, y - 1) if existence else 0

            features.extend([existence, x, y, lowest_value_attacker, lowest_value_defender, mobility])

        return features

    # Gets lowest attacker and defender for each square starting at A1 : pos = 0 , A2 : pos = 2, ... - 128 Features
    def get_square_centric_features(self, board):
        """
        Gets lowest attacker and defender for each square starting at A1 : pos = 0 , A2 : pos = 2, ...

        :param board: chess.Board
        """
        features = []

        for i in range(7, -1, -1):
            for j in range(8):
                lowest_attacker, lowest_defender = self.get_lowest_value_attacker_and_defender(board, j, i)
                features.extend([lowest_attacker, lowest_defender])

        return features

    def get_pos(self, x, y):
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

    def get_mobility(self, board, x, y):
        """
        :param board: chess.Board
        :param x: x pos (int)
        :param y: y pos (int)
        :return: get amount of spaces the piece on x,y can move
        """
        pos = self.get_pos(x, y)
        piece = board.piece_type_at(pos)
        mobility = 0
        positions = []  # integers for positions on board

        if piece == chess.PAWN:
            # Pawns at starting position
            if board.turn and y == 6:
                positions.append(self.get_pos(x, y - 2))
            if not board.turn and y == 1:
                positions.append(self.get_pos(x, y + 2))

            # Check diagonals and piece above it
            new_y = y - 1 if board.turn else y + 1
            new_pos = self.get_pos(x, new_y)
            positions.extend([new_pos-1, new_pos, new_pos+1])

        if piece == chess.KNIGHT:

            positions.extend([pos + 17, pos + 10,
                              pos - 6, pos - 15,
                              pos - 17, pos - 10,
                              pos + 6, pos + 15])

        if piece == chess.BISHOP:
            # Top left to bottom right \
            bottom = min(x, y)
            new_x, new_y = x - bottom, y - bottom
            for i in range(8-max(new_x, new_y)):
                positions.append(self.get_pos(new_x, new_y))
                new_x += 1
                new_y += 1

            # Bottom left to top right /
            top = max(x, y)
            new_x, new_y = x + top, y - top
            for i in range(max(new_x, 8 - new_y)):
                positions.append(self.get_pos(new_x, new_y))
                new_x -= 1
                new_y += 1

        if piece == chess.ROOK:
            # Vertical
            for i in range(8):
                positions.append(self.get_pos(x, i))

            # Horizontal
            for i in range(8):
                positions.append(self.get_pos(i, y))

        if piece == chess.QUEEN:
            # Vertical
            for i in range(8):
                positions.append(self.get_pos(x, i))

            # Horizontal
            for i in range(8):
                positions.append(self.get_pos(i, y))

            # Top Left to Bottom Right \
            bottom = min(x, y)
            new_x, new_y = x - bottom, y - bottom
            for i in range(8-max(new_x, new_y)):
                positions.append(self.get_pos(new_x, new_y))
                new_x += 1
                new_y += 1

            # Bottom left to top right /
            top = max(x, y)
            new_x, new_y = x + top, y - top
            for i in range(max(new_x, 8 - new_y)):
                positions.append(self.get_pos(new_x, new_y))
                new_x -= 1
                new_y += 1

        if piece == chess.KING:

            # Castling
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

            positions.extend([pos-1, pos+1])  # Left, Right
            # Top/Bottom
            new_y = y - 1
            new_pos = self.get_pos(x, new_y)
            positions.extend([new_pos-1, new_pos, new_pos+1])
            new_y = y + 1
            new_pos = self.get_pos(x, new_y)
            positions.extend([new_pos-1, new_pos, new_pos+1])

        for move in positions:
            try:
                if board.find_move(pos, move):
                    mobility += 1
            except:
                # board.find_move raises error if the move is not found
                pass

        return mobility

    # Gets lowest value attacker and defender of position x,y on board
    def get_lowest_value_attacker_and_defender(self, board, x, y):
        """
        :param board: chess.board
        :param x: x position of board (int)
        :param y: y position of board (int or char) -> char not tested
        :return: lowest_value_attacker, lower_value_defender
        """

        pos = self.get_pos(x, y)

        attackers_board = board.attackers(not board.turn, pos)
        defenders_board = board.attackers(board.turn, pos)

        lowest_value_attacker = 100
        lowest_value_defender = 100

        if len(attackers_board):
            for attacker_pos, square in enumerate(attackers_board.tolist()):
                if square:
                    lowest_value_attacker = min(lowest_value_attacker, board.piece_type_at(attacker_pos))
        else:
            lowest_value_attacker = 0

        if len(defenders_board):
            for defender_pos, square in enumerate(defenders_board.tolist()):
                if square:
                    lowest_value_defender = min(lowest_value_defender, board.piece_type_at(defender_pos))
        else:
            lowest_value_defender = 0

        return lowest_value_attacker, lowest_value_defender

    def get_piece_positions(self, board):
        """
        Creates a dictionary that stores the position of each piece as 'piece' : { 'x':(int),'y':(int) }
        Does not properly assign pieces to their slots
        -> Slots are filled by amount:
            - White pawn on e should be assigned to slot P5 but if eaten the slot will be taken by any extra pawns
            - Extra pieces will not be stored

        :param board: chess.Board()
        :return: dict with positions of each piece, 'piece' as key
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
                        if piece in positions:
                            positions[piece]['x'] = x + 1  # +1 because 0 represents not there
                            positions[piece]['y'] = y + 1
                    else:
                        slot = 1

                        piece_slot = piece + str(slot)
                        while positions[piece_slot]['x'] != 0:
                            slot += 1
                            piece_slot = piece + str(slot)

                            if piece_slot not in positions:
                                break

                        if piece_slot in positions:
                            positions[piece_slot]['x'] = x + 1
                            positions[piece_slot]['y'] = y + 1

        return positions
