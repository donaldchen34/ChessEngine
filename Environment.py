import chess
import chess.svg
import random
import time
from PyQt5.Qt import pyqtSignal, QThread
from Computer import Computer
from BoardRepresentation import Evaluator, convertBoardToList


# Todo
# Quality of Life:
# Player Moves:
# - Cant switch pieces after choosing one
# - Create indicator for what piece is clicked
# - Flip board according to player piece


class Environment(QThread):
    update_board_signal = pyqtSignal()

    def __init__(self,update_board_func ):
        super(Environment, self).__init__()
        self.board = chess.Board()
        self.turn = random.randint(0,1)  # 0 - Player, 1 - Computer

        self.update_board_func = update_board_func

        self.piece_selected = False
        self.piece_selected_pos = -1
        self.queue = []

        self.evaluator = Evaluator()
        self.computer = Computer(board=self.board, algo='dqn')

    def show_board(self):
        print(self.board)
        print('----------------')

    # Might be missing some conditions
    def game_over(self):
        return self.board.is_checkmate() or self.board.is_game_over() or self.board.is_stalemate()

    def makePlayerMove(self,x,y):

        ROWS = 8
        if x < ROWS and y < ROWS and x >= 0 and y >= 0:
            piece = convertBoardToList(self.board)[y][x]

            pos = x + ((ROWS - 1) - y) * ROWS  # Check docs chess.Squares -> chess.A1 = 0 ... chess.H8 = 63
            # If no piece is currently selected
            if not self.piece_selected:
                if piece != '.':  # If not empty space

                    self.piece_selected_pos = pos
                    self.piece_selected = piece
            else:  # Piece is selected
                move = chess.Move(from_square=self.piece_selected_pos, to_square=pos)

                # Promotion
                if self.piece_selected == 'p' and y == ROWS - 1 or self.piece_selected == 'P' and y == 0:
                    move = chess.Move(from_square=self.piece_selected_pos, to_square=pos, promotion=5)

                # If valid move
                if self.getBoard().is_legal(move):
                    self.queue.append(move)
                # If not valid move
                else:
                    self.update_board_signal.emit()

                self.piece_selected = False

    def player_turn(self):
        if len(self.queue) == 0:
            while len(self.queue) == 0:
                time.sleep(0.5)

        player_move = self.queue.pop()
        self.board.push(player_move)

    def computer_turn(self):
        self.computer.make_move()

    def play_game(self):
        while not self.game_over():
            # 0 - Player, 1 - Computer
            if self.turn % 2 == 0:
                self.player_turn()

            if self.turn % 2 == 1:
                self.computer_turn()

            print("Eval:", self.basicEvaluation())
            self.turn += 1
            self.update_board_signal.emit()

        print("Game Over")

    def self_play(self):
        while not self.game_over():
            print("{} Turn".format("White" if self.turn % 2 == 0 else "Black"))
            self.computer_turn()
            self.turn += 1
            self.update_board_signal.emit()
            time.sleep(2)

    def newGame(self):
        self.board.reset()
        self.turn = random.randint(0,1) #0 - Player, 1 - Computer

    def run(self):
        print("You are {}".format("White" if self.turn == 0 else "Black"))

        print("PLAYING")
        self.play_game()
        #self.self_play()

    def getBoard(self):
        return self.board

    def basicEvaluation(self):
        return self.evaluator.getEval(board=self.board, turn_count=self.turn, turn=self.board.turn)



if __name__ == "__main__":
    game = Environment()
    game.show_board()