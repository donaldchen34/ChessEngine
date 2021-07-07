import chess
import chess.svg
import random
import time
from PyQt5.Qt import pyqtSignal, QThread
from Computer import Computer
from BoardRepresentation import Evaluator, convertBoardToList

#Todo
#makePlayerMove():
# - Cant switch pieces after choosing one
# - Sometimes Breaks
# - Indicator for what piece is clicked
# - Handle flipped board(if player has black pieces)


class Environment(QThread):
    update_board_signal = pyqtSignal()

    def __init__(self,update_board_func ):
        super(Environment, self).__init__()
        self.board = chess.Board()
        self.turn = random.randint(0,1) #0 - Player, 1 - Computer

        self.update_board_func = update_board_func #Used for attack squares -> Need to change later -> Maybe a diff emit?

        self.piece_selected = False
        self.piece_selected_pos = -1
        self.queue = []

        self.evaluator = Evaluator()
        self.computer = Computer(board=self.board)

    def showBoard(self):
        print(self.board)
        print('----------------')

    #Might be missing some conditions
    def gameOver(self):
        return self.board.is_checkmate() or self.board.is_game_over() or self.board.is_stalemate()

    def makePlayerMove(self,x,y):

        ROWS = 8
        if x < ROWS and y < ROWS and x >= 0 and y >= 0:
            piece = convertBoardToList(self.board)[y][x]

            pos = x + ((ROWS - 1) - y) * ROWS  # Check docs chess.Squares -> chess.A1 = 0 ... chess.H8 = 63
            # If no piece is currently selected
            if not self.piece_selected:
                if piece != '.':  # If not empty space

                    # Change to spaces that can be moved?
                    #squares = self.getBoard().attacks(pos)
                    #self.selected_piece_signal.emit(chess.svg.board(self.getBoard(), squares=squares, size=900))

                    squares1 = self.getBoard().is_attacked_by(not self.board.turn, pos)
                    squares2 = self.getBoard().attacks(pos)
                    squares3 = self.getBoard().attackers(not self.board.turn, pos)

                    print(len(squares3))
                    temp = []
                    if len(squares3):
                        for attacker_pos, square in enumerate(squares3.tolist()):
                            if square:
                                print(attacker_pos,square)
                                print(squares3)
                                print(squares3.piece_at(attacker_pos))
                                temp.append(squares3.piece_type_at(attacker_pos))

                    print(temp)

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

    def playerTurn(self):
        if len(self.queue) == 0:
            while (len(self.queue) == 0):
                time.sleep(0.5)

        player_move = self.queue.pop()
        self.board.push(player_move)

    def computerTurn(self):
        #Random move
        #moves = [move for move in self.board.legal_moves]
        #comp_move = random.randint(0, len(moves) - 1)
        #self.board.push(moves[comp_move]
        self.computer.makeMove()

    def playGame(self):
        while (not self.gameOver()):
            # 0 - Player, 1 - Computer
            if self.turn % 2 == 0:
                self.playerTurn()

            if self.turn % 2 == 1:
                self.computerTurn()

            print("Eval:", self.basicEvaluation())
            self.turn += 1
            self.update_board_signal.emit()

        print("Game Over")

    def newGame(self):
        self.board.reset()
        self.turn = random.randint(0,1) #0 - Player, 1 - Computer

    def run(self):
        print("You are {}".format("White" if self.turn == 0 else "Black"))

        print("PLAYING")
        self.playGame()


    def getBoard(self):
        return self.board

    def basicEvaluation(self):
        Board_list = convertBoardToList(board=self.board)
        return self.evaluator.getEval(board=Board_list,turn_count=self.turn,turn=self.board.turn)


if __name__ == "__main__":
    game = Environment()
    game.showBoard()