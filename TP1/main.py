import soko
import sys
from State import State
from tree import Tree, Node
import algorithms
import utils
import time

# Definir el tamaño de la ventana y el tamaño de la celda
ANCHO = 550
ALTO = 550
TAMANO_CELDA = 30


def cargar_niveles (ruta_archivo):
    """
    Recibe la ruta de un archivo y devuelve una lista de listas, donde cada sublista es un nivel
    """
    max_longitud_fila = 0
    
    with open (ruta_archivo) as archivo:
        total_niveles = archivo.readlines()
        nivel = []
        niveles = []
        
        for fila in total_niveles:
            fila = fila[:-1]
            
            if not "#" in fila and fila != "":
                continue
            
            if fila == "":
                for i in range (len(nivel)):
                    nivel[i] += (" " * (max_longitud_fila - len(nivel[i])))
                
                niveles.append(nivel)
                nivel = []
                max_longitud_fila = 0
                continue
            
            if len(fila) > max_longitud_fila:
                max_longitud_fila = len(fila)
                
            nivel.append(fila)
        
        return niveles


def start(*args):
    if args:
        level, method, heuristic = args
    else:
        # Si no se pasan argumentos directamente, lee los argumentos de la línea de comandos
        parsed_args = utils.readCommand(sys.argv)
        level, method, heuristic = parsed_args.values()

    heuristicFunction = algorithms.manhattan_heuristic
    if heuristic == "combined":
        heuristicFunction = algorithms.combined_heuristic

    levels = cargar_niveles("niveles.txt")
    board = soko.crear_grilla(levels[level])

    (boardMatrix, playerPos, goalsPos, boxesPos) = utils.sanitize_level(board)

    initialState = State(playerPos, boxesPos, goalsPos)

    # Registrar el tiempo de inicio
    start = time.time()

    path = []
    cost = 0
    exploredNodes = []
    frontierNodes = []
    if method == 'bfs':
        path, cost, exploredNodes, frontierNodes = algorithms.bfs(initialState, boardMatrix)
    elif method == 'dfs':
        path, cost, exploredNodes, frontierNodes = algorithms.dfs(initialState, boardMatrix)
    elif method == 'greedy':
        path, cost, exploredNodes, frontierNodes = algorithms.greedy(initialState, boardMatrix, heuristicFunction)
    elif method == 'astar':
        path, cost, exploredNodes, frontierNodes = algorithms.astar(initialState, boardMatrix, heuristicFunction)

    # Registrar el tiempo de finalización
    end = time.time()

    original_stdout = sys.stdout

    # Abrir un archivo para escribir
    with open('output.txt', 'w') as f:
        # Redirigir la salida estándar al archivo
        sys.stdout = f
        for node in path:
            node.state.print_board(soko.regenerate(board, node.state.playerPos, node.state.goalsPos, node.state.boxesPos))
            print()
    sys.stdout = original_stdout

    delta = end - start

    return {'path': path, 'level': level, 'method': method, 'heuristic': heuristic, 'cost': cost, 'exploredNodes': exploredNodes, 'frontierNodes': frontierNodes, 'delta': delta}






if __name__ == "__main__":
    results = start()

    print('\nResultados' if results['path'] != 1 else '\nResult: Couldn\'t find solution\n')
    print('Se recorrio el nivel: ' + str(results['level']))
    print('Se utilizo el metodo: ' + results['method'])
    if results['method'] == 'greedy' or results['method'] == 'astar':
        print('Con la heuristica: ' + results['heuristic'])
    print('El costo de la solución fue: ' + str(results['cost']))
    print('Los nodos explorados fueron: ' + str(results['exploredNodes']))
    print('La cantidad de nodos fueron: ' + str(results['frontierNodes']))
    print('El tiempo fue: %.3f' % results['delta'] + ' seg')
