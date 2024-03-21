


class State:
    
    def __init__(self, playerPos:tuple, boxesPos:list[tuple], goalsPos: list[tuple]):
        self.playerPos = playerPos
        self.boxesPos = boxesPos
        self.goalsPos = goalsPos

    def is_finished(self) -> bool:
        return self.boxesPos == self.goalsPos