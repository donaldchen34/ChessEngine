import chess
import chess.svg
import random

class Environment:
    def __init__(self):
        self.board = chess.Board()
        self.turn = random.randint(0,1) #0 - Player, 1 - Computer

        self.run()

    def showBoard(self):
        print(self.board) #Change to GUI
        print(chess.svg.board(self.board))
        print('----------------')

    def gameOver(self):
        return self.board.is_checkmate()

    def playGame(self):
        # 0 - Player, 1 - Computer
        self.showBoard()
        if self.turn == 0:
            print("Player Turn")
            print("Your possible moves:")
            print(self.board.legal_moves)
            player_move = input("Enter your move: ")
            self.board.push_san(player_move)

            self.turn += 1

        if self.turn == 1:
            #Currently Computer makes a random move
            #Need to change to RL or Minmax (More intelligent)
            print("Computer turn")
            moves = [move for move in self.board.legal_moves]
            comp_move = random.randint(0,len(moves))
            self.board.push(moves[comp_move])

            self.turn -= 1

    def newGame(self):
        pass

    def run(self):
        print("You are {}".format("White" if self.turn == 0 else "Black"))
        while (not self.gameOver()):
            self.playGame()


if __name__ == "__main__":
    game = Environment()
    game.showBoard()