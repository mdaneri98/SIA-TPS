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


def main():
    # Inicializar el estado del juego
    actual_level = 1

    levels = cargar_niveles("niveles.txt")
    level = soko.crear_grilla(levels[actual_level])
    
    (level, playerPos, goalsPos, boxesPos) = utils.sanitize_level(level)

    state = State(playerPos, boxesPos, goalsPos)
    
    # Registrar el tiempo de inicio
    start = time.time()

    (success, cost, nodes_count, frontier_nodes) = algorithms.bfs(state, level)

    for node in success:
        node.state.print_board(soko.regenerate(level, node.state.playerPos, node.state.goalsPos, node.state.boxesPos))
        print()

  #  print((success, cost, nodes_count, frontier_nodes))

    # Registrar el tiempo de finalización
    end = time.time()
    delta = end - start

   
if __name__ == "__main__":
    main()
