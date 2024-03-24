import os
from optparse import OptionParser

SEARCH_METHODS = ['bfs', 'dfs', 'greedy', 'astar']
HEURISTICS = ['manhattan', 'combined']
def readCommand(argv):
    parser = OptionParser()
    parser.add_option('-l', '--level', dest='level', help='level of game to play', default='2', type="int")
    parser.add_option('-m', '--method', dest='method', help='research method', default='astar')
    parser.add_option('-H', '--heuristic', dest='heuristic', help='heuristic', default='manhattan')
    args = dict()
    options, _ = parser.parse_args(argv)

    if options.method not in SEARCH_METHODS:
        print("Choose a supported research method")
        exit(1)
    if options.heuristic not in HEURISTICS:
        print("Choose a supported heuristic")
        exit(1)

    args['level'] = options.level
    args['method'] = options.method
    args['heuristic'] = options.heuristic
    return args


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
