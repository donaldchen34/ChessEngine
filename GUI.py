from Environment import Environment
import chess
import chess.svg
from PyQt5.QtSvg import QSvgWidget
from PyQt5.QtWidgets import QApplication, QWidget
from math import floor

# Todo
# Add restart Game Button
# Add option to choose which computer mode to play

class GUI(QWidget):

    def __init__(self, size=900):
        super().__init__()

        self.size = size

        # https://stackoverflow.com/questions/61439815/how-to-display-an-svg-image-in-python
        self.setGeometry(100, 100, self.size, self.size)
        self.widgetSvg = QSvgWidget(parent=self)
        self.widgetSvg.setGeometry(0, 0, self.size, self.size)

        self.setMouseTracking(True)

        self.env = Environment(update_board_func=self.load_board)
        self.env.update_board_signal.connect(self.load_board)
        try:
            self.env.start()
        except Exception as e:
            print(e)

        self.load_board()

    def load_board(self, board=None):
        self.chessboardSvg = chess.svg.board(self.env.get_board()).encode("UTF-8") if board == None else board.encode("UTF-8")
        self.widgetSvg.load(self.chessboardSvg)

    def mousePressEvent(self, event):
        # Borders: 35, 865
        BORDER_LEN = 35
        ROWS = 8

        x = floor((event.x() - BORDER_LEN) / 100)
        y = floor((event.y() - BORDER_LEN) / 100)

        if x < ROWS and y < ROWS and x >= 0 and y >= 0:
            self.env.make_player_move(x, y)


if __name__ == "__main__":
    app = QApplication([])
    window = GUI()
    window.show()
    app.exec()
