# https://www.chessprogramming.org/Simplified_Evaluation_Function
# https://chess.stackexchange.com/questions/347/what-is-an-accurate-way-to-evaluate-chess-positions
# https://www.chessprogramming.org/Evaluation
# https://www.cmpe.boun.edu.tr/~gungort/undergraduateprojects/Tuning%20of%20Chess%20Evaluation%20Function%20by%20Using%20Genetic%20Algorithms.pdf
# 4.1 ^

# https://chess.stackexchange.com/questions/347/what-is-an-accurate-way-to-evaluate-chess-positions
# Example of eval function
# c1 * material + c2 * mobility + c3 * king safety + c4 * center control + c5 * pawn structure + c6 * king tropism + ...

PIECE_VALUE = {
    'Q': 90,  # Queen
    'R': 50,  # Rook
    'N': 32,  # Knight
    'B': 33,  # Bishop
    'P': 10,  # Pawn
    'K': 2000  # King
}

SQUARE_TABLES = {
    "P": [0,  0,  0,  0,  0,  0,  0,  0,
          50, 50, 50, 50, 50, 50, 50, 50,
          10, 10, 20, 30, 30, 20, 10, 10,
          5,  5, 10, 25, 25, 10,  5,  5,
          0,  0,  0, 20, 20,  0,  0,  0,
          5, -5,-10,  0,  0,-10, -5,  5,
          5, 10, 10,-20,-20, 10, 10,  5,
          0,  0,  0,  0,  0,  0,  0,  0],
    "N": [-50,-40,-30,-30,-30,-30,-40,-50,
          -40,-20,  0,  0,  0,  0,-20,-40,
          -30,  0, 10, 15, 15, 10,  0,-30,
          -30,  5, 15, 20, 20, 15,  5,-30,
           30,  0, 15, 20, 20, 15,  0,-30,
          -30,  5, 10, 15, 15, 10,  5,-30,
          -40,-20,  0,  5,  5,  0,-20,-40,
          -50,-40,-30,-30,-30,-30,-40,-50],
    "B": [-20,-10,-10,-10,-10,-10,-10,-20,
          -10,  0,  0,  0,  0,  0,  0,-10,
          -10,  0,  5, 10, 10,  5,  0,-10,
          -10,  5,  5, 10, 10,  5,  5,-10,
          -10,  0, 10, 10, 10, 10,  0,-10,
          -10, 10, 10, 10, 10, 10, 10,-10,
          -10,  5,  0,  0,  0,  0,  5,-10,
          -20,-10,-10,-10,-10,-10,-10,-20],
    "R": [-20,-10,-10,-10,-10,-10,-10,-20,
          -10,  0,  0,  0,  0,  0,  0,-10,
          -10,  0,  5, 10, 10,  5,  0,-10,
          -10,  5,  5, 10, 10,  5,  5,-10,
          -10,  0, 10, 10, 10, 10,  0,-10,
          -10, 10, 10, 10, 10, 10, 10,-10,
          -10,  5,  0,  0,  0,  0,  5,-10,
          -20,-10,-10,-10,-10,-10,-10,-20],
    "Q": [-20,-10,-10, -5, -5,-10,-10,-20,
          -10,  0,  0,  0,  0,  0,  0,-10,
          -10,  0,  5,  5,  5,  5,  0,-10,
          -5,  0,  5,  5,  5,  5,  0, -5,
           0,  0,  5,  5,  5,  5,  0, -5,
          -10,  5,  5,  5,  5,  5, 0, -10,
          -10,  0,  5,  0,  0,  0, 0, -10,
          -20,-10,-10, -5, -5,-10,-10,-20],
    'K': {
            "MIDDLE_GAME": [-30,-40,-40,-50,-50,-40,-40,-30,
                            -30,-40,-40,-50,-50,-40,-40,-30,
                            -30,-40,-40,-50,-50,-40,-40,-30,
                            -30,-40,-40,-50,-50,-40,-40,-30,
                            -20,-30,-30,-40,-40,-30,-30,-20,
                            -10,-20,-20,-20,-20,-20,-20,-10,
                             20, 20,  0,  0,  0,  0, 20, 20,
                             20, 30, 10,  0,  0, 10, 30, 20],
            "END_GAME": [-50,-40,-30,-20,-20,-30,-40,-50,
                         -30,-20,-10,  0,  0,-10,-20,-30,
                         -30,-10, 20, 30, 30, 20,-10,-30,
                         -30,-10, 30, 40, 40, 30,-10,-30,
                         -30,-10, 30, 40, 40, 30,-10,-30,
                         -30,-10, 20, 30, 30, 20,-10,-30,
                         -30,-30,  0,  0,  0,  0,-30,-30,
                         -50,-30,-30,-30,-30,-30,-30,-50]
    }
}


# https://stackoverflow.com/questions/55876336/is-there-a-way-to-convert-a-python-chess-board-into-a-list-of-integers
def convert_board_to_list(board):
    board_list = []
    temp = board.epd()

    pieces = temp.split(" ", 1)[0]
    rows = pieces.split("/")
    for row in rows:
        temp2 = []  # This is the row I make
        for thing in row:
            if thing.isdigit():
                for i in range(0, int(thing)):
                    temp2.append('.')
            else:
                temp2.append(thing)
        board_list.append(temp2)
    return board_list


class Evaluator:
    def get_eval(self, board, turn_count, turn):

        self.board = board  # chess.Board()
        self.board_list = convert_board_to_list(board)

        self.turn_count = turn_count  # Amount of turns
        self.turn = turn  # Black or White to move

        # Missing:
        # Mobility
        # Pawn Structure (Double,Blocked, Isolated)
        # Attacked by/ Defended by
        # Turn Bonus - Whose move it is
        # Special Conditions:
        # King Pawn Tropism
        # Castling
        # 2 Bishops
        # Checks/ Checkmate

        if self.is_game_over():
            outcome = self.board.outcome()
            if self.board.is_stalemate():
                pass
            if outcome.winner == turn:
                return 100000
            else:
                return -100000

        total_score = 0

        # Materials
        for x, row in enumerate(self.board_list):
            for y, piece in enumerate(row):
                if piece != '.':
                    if piece.isupper():
                        total_score += self.get_piece_value(piece)
                        total_score += self.get_position_bonus(piece, x, y)
                    if piece.islower():
                        total_score -= self.get_piece_value(piece)
                        total_score -= self.get_position_bonus(piece, x, y)

        total_score += self.get_turn_bonus()

        # Change to a percentage?
        return total_score

    def is_game_over(self):
        return self.board.is_game_over()
        # return self.board.is_checkmate() or self.board.is_game_over() or self.board.is_stalemate()

    def get_turn_bonus(self):
        # True - White Turn, False - Black Turn
        return 30 if self.turn else -30

    def get_game_state(self):
        # Currently no Opening Square Table for King
        # if self.turn_count < 10:
        #    return "OPENING"
        if self.turn_count < 60:
            return "MIDDLE_GAME"
        else:
            return "END_GAME"

    def get_position_bonus(self, piece, x, y):

        if piece.islower():  # For black pieces, SQUARE_TABLES is white orientated
            y = 7 - y
            x = 7 - x

        pos = x * 8 + y
        piece = piece.upper()

        if piece == 'K':
            return SQUARE_TABLES[piece][self.get_game_state()][pos]
        else:
            return SQUARE_TABLES[piece][pos]

    def get_piece_value(self, piece):
        piece = piece.upper()
        return PIECE_VALUE[piece]
