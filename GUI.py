from Environment import Environment
import chess
import chess.svg
from PyQt5.QtSvg import QSvgWidget
from PyQt5.QtWidgets import QApplication, QWidget
from math import floor

#https://stackoverflow.com/questions/61439815/how-to-display-an-svg-image-in-python
class GUI(QWidget):
    def __init__(self, size = 900):
        super().__init__()


        self.size = size
        self.env = Environment()

        self.setGeometry(100, 100, self.size, self.size)
        self.widgetSvg = QSvgWidget(parent=self)
        self.widgetSvg.setGeometry(0, 0, self.size, self.size)

        self.setMouseTracking(True)
        self.loadBoard()
        self.piece_selected = False
        self.piece_selected_pos = -1

    def loadBoard(self, board = None):
        self.chessboardSvg = chess.svg.board(self.env.getBoard()).encode("UTF-8") if board == None else board.encode("UTF-8")
        self.widgetSvg.load(self.chessboardSvg)

    def playGame(self):
        pass

    def mousePressEvent(self, event):
        # Borders: 35,865
        BORDER_LEN = 35
        ROWS = 8
        print(event.x(),event.y())

        x = floor((event.x() - BORDER_LEN) / 100)
        y = floor((event.y() - BORDER_LEN) / 100)
        print("COORDS:", x,y)
        #No piece is selected
        if x < 8 and y < 8 and x > 0 and y > 0:

            y = (7 - y) * 8
            pos = x + y
            piece = self.env.convertBoardToList()[y][x]

            #If no piece is selected and selected space is not empty
            if not self.piece_selected and piece != '.':

                #squares = self.env.getBoard().attacks(pos)
                #self.loadBoard(chess.svg.board(self.env.getBoard(),squares=squares,size = self.size))

                self.piece_selected_pos = pos
                self.piece_selected = True

            else:
                y = (7 - y) * 8
                pos = x + y
                #If valid move
                if self.env.getBoard().is_legal(chess.Move(from_square=self.piece_selected_pos,to_square=pos)):
                    print("MOVE")
                #If not valid move
                else:
                    print("Bad Move")


                self.piece_selected = False


if __name__ == "__main__":
    app = QApplication([])
    window = GUI()
    window.show()
    app.exec()