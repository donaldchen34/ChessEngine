import chess
import chess.svg
import random

class Environment:
    def __init__(self):
        self.board = chess.Board()
        self.turn = random.randint(0,1) #0 - Player, 1 - Computer

        #self.run()

    def showBoard(self):
        print(self.board) #Change to GUI
        print('----------------')

    def gameOver(self):
        return self.board.is_checkmate()

    def playerTurn(self):
        pass

    def computerTurn(self):
        pass

    def playGame(self):
        # 0 - Player, 1 - Computer
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
            comp_move = random.randint(0,len(moves)-1)
            print(comp_move)
            print(len(moves))
            self.board.push(moves[comp_move])

            self.turn -= 1

    def newGame(self):
        pass

    def run(self):
        print("You are {}".format("White" if self.turn == 0 else "Black"))
        while (not self.gameOver()):
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