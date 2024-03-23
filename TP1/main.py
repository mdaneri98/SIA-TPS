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
    print(level)
    
    (level, playerPos, goalsPos, boxesPos) = utils.sanitize_level(level)

    print("mapa sanitizado: \nplayerPos: {}\ngoalsPos:{}\nboxesPos:{}\n".format(playerPos, goalsPos, boxesPos))
    print(level)
    print()

    state = State(playerPos, boxesPos, goalsPos)
    
    # Registrar el tiempo de inicio
    start = time.time()

    print("mapa regenarizado: \nplayerPos: {}\ngoalsPos:{}\nboxesPos:{}\n".format(playerPos, goalsPos, boxesPos))
    print(state.print_board(soko.regenerate(level, state.playerPos, state.goalsPos, state.boxesPos)))
    print()

    #(success, cost, nodes_count, frontier_nodes) = algorithms.bfs(state, level)
    #print((success, cost, nodes_count, frontier_nodes))

    # Registrar el tiempo de finalización
    end = time.time()
    delta = end - start

   

if __name__ == "__main__":
    main()
