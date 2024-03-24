class State(object):

    def __init__(self, playerPos: tuple, boxesPos: list[tuple], goalsPos: list[tuple]):
        self.playerPos = playerPos
        self.boxesPos = boxesPos
        self.goalsPos = goalsPos

    def __eq__(self, other):
        result = self.__class__ == other.__class__ and self.playerPos == other.playerPos and self.boxesPos == other.boxesPos
        return result

    def __hash__(self):
        return hash((self.playerPos, tuple(self.boxesPos)))

    def is_finished(self) -> bool:
        return self.boxesPos == self.goalsPos

    def print_board(self, board):
        if board is None:
            print("No hay tablero disponible para imprimir.")
            return

        for i, row in enumerate(board):
            for j, cell in enumerate(row):
                if (i, j) == self.playerPos:
                    print("@", end="")
                elif (i, j) in self.boxesPos:
                    print("$", end="")
                elif (i, j) in self.goalsPos:
                    print(".", end="")
                elif board[i][j] == "#":
                    print("#", end="")
                else:
                    print(cell, end="")
            print()
