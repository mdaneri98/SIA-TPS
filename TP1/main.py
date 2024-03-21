from pila import Pila
from cola import Cola
import soko
import sys

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
    pila_aux = Pila()
    cola_aux = Cola()
    nivel_actual = 0

    niveles = cargar_niveles ("niveles.txt")
    nivel = soko.crear_grilla(niveles[nivel_actual])
    
    while True:
        direccion = read_input()
        nivel = soko.mover (nivel, direccion)

        #Chequea el fin del nivel y pasa al siguiente
        if soko.juego_ganado (nivel):
            nivel_actual += 1
                
            if nivel_actual == len(niveles):
                return
                
            nivel = soko.crear_grilla(niveles[nivel_actual])
                

       

if __name__ == "__main__":
    main()
