from Environment import Environment
import chess
import chess.svg
from PyQt5.QtSvg import QSvgWidget
from PyQt5.QtWidgets import QApplication, QWidget
from math import floor
from PyQt5.Qt import pyqtSignal, QThread

#https://stackoverflow.com/questions/61439815/how-to-display-an-svg-image-in-python
#https://stackoverflow.com/questions/52993677/how-do-i-setup-signals-and-slots-in-pyqt-with-qthreads-in-both-directions
class GUI(QWidget):
    update_board = pyqtSignal()

    def __init__(self, size = 900):
        super().__init__()


        self.size = size
        self.setGeometry(100, 100, self.size, self.size)
        self.widgetSvg = QSvgWidget(parent=self)
        self.widgetSvg.setGeometry(0, 0, self.size, self.size)

        self.setMouseTracking(True)

        self.piece_selected = False
        self.piece_selected_pos = -1
        self.moveQueue = []

        self.env = Environment(signal = self.update_board,queue=self.moveQueue)
        thread = QThread(self)
        self.env.moveToThread(thread)
        thread.start()

        self.loadBoard()

        self.update_board.connect(self.loadBoard)

    def loadBoard(self, board = None):
        self.chessboardSvg = chess.svg.board(self.env.getBoard()).encode("UTF-8") if board == None else board.encode("UTF-8")
        self.widgetSvg.load(self.chessboardSvg)

    def playGame(self):
        print("You are {}".format("White" if self.env.turn == 0 else "Black"))
        while(not self.env.gameOver()):
            print("PLAYING")
            if len(self.updateBoardNotice):
                self.updateBoardNotice.pop()
                self.loadBoard()






    #Cant switch pieces after choosing one
    def mousePressEvent(self, event):
        # Borders: 35,865
        BORDER_LEN = 35
        ROWS = 8
        print(event.x(),event.y())

        x = floor((event.x() - BORDER_LEN) / 100)
        y = floor((event.y() - BORDER_LEN) / 100)
        print("COORDS:", x,y)
        #No piece is selected
        if x < 8 and y < 8 and x >= 0 and y >= 0:

            piece = self.env.convertBoardToList()[y][x]

            y = (7 - y) * 8
            pos = x + y


            #If no piece is currenlt selected
            if not self.piece_selected:
                if piece != '.': #If not empty space

                    #Change to spaces that can be moved?
                    squares = self.env.getBoard().attacks(pos)
                    self.loadBoard(chess.svg.board(self.env.getBoard(),squares=squares,size = self.size))

                    self.piece_selected_pos = pos
                    self.piece_selected = True
                if piece == '.':
                    print("Empty Space")
            else: #Piece is selected
                move = chess.Move(from_square=self.piece_selected_pos,to_square=pos)
                #If valid move
                if self.env.getBoard().is_legal(move):
                    print("MOVE")
                    self.moveQueue.append(move)
                    print("Move:", move)
                    print(self.moveQueue)
                #If not valid move
                else:
                    print("Bad Move")
                    self.loadBoard()

                self.piece_selected = False


if __name__ == "__main__":
    app = QApplication([])
    window = GUI()
    window.show()
    app.exec()