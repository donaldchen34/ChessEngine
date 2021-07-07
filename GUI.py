from Environment import Environment
import chess
import chess.svg
from PyQt5.QtSvg import QSvgWidget
from PyQt5.QtWidgets import QApplication, QWidget
from math import floor

#https://stackoverflow.com/questions/61439815/how-to-display-an-svg-image-in-python
#https://stackoverflow.com/questions/52993677/how-do-i-setup-signals-and-slots-in-pyqt-with-qthreads-in-both-directions
class GUI(QWidget):

    #Add restart Game Button
    def __init__(self, size = 900):
        super().__init__()

        self.size = size
        self.setGeometry(100, 100, self.size, self.size)
        self.widgetSvg = QSvgWidget(parent=self)
        self.widgetSvg.setGeometry(0, 0, self.size, self.size)

        self.setMouseTracking(True)

        self.env = Environment(update_board_func = self.loadBoard)
        self.env.update_board_signal.connect(self.loadBoard)
        try:
            self.env.start()
        except Exception as e:
            print(e)

        self.loadBoard()

    def loadBoard(self, board = None):
        self.chessboardSvg = chess.svg.board(self.env.getBoard()).encode("UTF-8") if board == None else board.encode("UTF-8")
        self.widgetSvg.load(self.chessboardSvg)

    def mousePressEvent(self, event):
        # Borders: 35,865
        BORDER_LEN = 35
        ROWS = 8

        x = floor((event.x() - BORDER_LEN) / 100)
        y = floor((event.y() - BORDER_LEN) / 100)

        if x < ROWS and y < ROWS and x >= 0 and y >= 0:
            self.env.makePlayerMove(x,y)



if __name__ == "__main__":
    app = QApplication([])
    window = GUI()
    window.show()
    app.exec()
