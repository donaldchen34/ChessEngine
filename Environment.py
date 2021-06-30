import chess
import chess.svg
import random
import time
from PyQt5.Qt import pyqtSignal, QThread

#Todo
#makePlayerMove():
# - Cant switch pieces after choosing one
# - Sometimes Breaks
# - Indicator for what piece is clicked
# - Handle flipped board(if player has black pieces)
#computerMove():
# - Make intelligent


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

    def showBoard(self):
        print(self.board)
        print('----------------')

    def gameOver(self):
        return self.board.is_checkmate() or self.board.is_game_over() or self.board.is_stalemate()

    def makePlayerMove(self,x,y):

        ROWS = 8
        if x < ROWS and y < ROWS and x >= 0 and y >= 0:
            piece = self.convertBoardToList()[y][x]

            pos = x + ((ROWS - 1) - y) * ROWS  # Check docs chess.Squares -> chess.A1 = 0 ... chess.H8 = 63
            # If no piece is currently selected
            if not self.piece_selected:
                if piece != '.':  # If not empty space

                    # Change to spaces that can be moved?
                    #squares = self.getBoard().attacks(pos)
                    #self.selected_piece_signal.emit(chess.svg.board(self.getBoard(), squares=squares, size=900))

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
        # Currently Computer makes a random move
        # Need to change to RL or Minmax (More intelligent)
        moves = [move for move in self.board.legal_moves]
        comp_move = random.randint(0, len(moves) - 1)
        self.board.push(moves[comp_move])

    def playGame(self):
        while (not self.gameOver()):
            # 0 - Player, 1 - Computer
            if self.turn % 2 == 0:
                self.playerTurn()

            if self.turn % 2 == 1:
                self.computerTurn()

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

    #https://stackoverflow.com/questions/55876336/is-there-a-way-to-convert-a-python-chess-board-into-a-list-of-integers
    def convertBoardToList(self):
        Board_list = []
        temp = self.board.epd()

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
            Board_list.append(temp2)
        return Board_list



if __name__ == "__main__":
    game = Environment()
    game.showBoard()