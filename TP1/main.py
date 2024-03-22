import soko
import sys
import State
from tree import Tree, Node
import algorithms

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

def read_input():
    move = input("Enter your move (wasd): ").lower()

    if move == 'w':
        return (0, -1)
    elif move == 's':
        return (0, 1)
    elif move == 'a':
        return (-1, 0)
    elif move == 'd':
        return (1, 0)
    else:
        return (0,0)

def main():
    # Inicializar el estado del juego
    nivel_actual = 1

    niveles = cargar_niveles ("niveles.txt")
    nivel = soko.crear_grilla(niveles[nivel_actual])
    
    playerPos
    goalsPos = []
    boxesPos = []
    for i, _ in nivel:
        for j in len(nivel):
            if (nivel[i][j] == '$'):
                boxesPos.append((i,j))
            elif nivel[i][j] == '@':
                playerPos = (i,j)
            elif nivel[i][j] == '.':
                goalsPos.append((i,j))

 
    state = State(nivel, playerPos, boxesPos, goalsPos)
    tree = Tree(state)
 
    algorithms.bfs(tree)









if __name__ == "__main__":
    main()
