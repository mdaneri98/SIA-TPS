


def sanitize_level(level):
    playerPos = None
    goalsPos = []
    boxesPos = []
    for i in range(len(level)):
        for j in range(len(level[i])):
            if level[i][j] == '$':
                level[i][j] = ' '
                boxesPos.append((i,j))
            elif level[i][j] == '@':
                level[i][j] = ' '
                playerPos = (i,j)
            elif level[i][j] == '.':
                level[i][j] = ' '
                goalsPos.append((i,j))
    return (level, playerPos, goalsPos, boxesPos)
