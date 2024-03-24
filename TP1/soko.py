import copy


PARED = "#"
CAJA = "$"
JUGADOR = "@"
OBJETIVO = "."
ESPACIO_VACIO = " "

OESTE = (-1, 0)
ESTE = (1, 0)
NORTE = (0, -1)
SUR = (0, 1)

def crear_grilla(desc):
    '''Crea una grilla a partir de la descripción del estado inicial.

    La descripción es una lista de cadenas, cada cadena representa una
    fila y cada caracter una celda. Los caracteres pueden ser los siguientes:

    Caracter  Contenido de la celda
    --------  ---------------------
           #  Pared
           $  Caja
           @  Jugador
           .  Objetivo
           *  Objetivo + Caja
           +  Objetivo + Jugador
    '''
    
    grilla = []
    
    for f in range (len(desc)):
        fila = []
        for c in range(len(desc[f])):
            fila.append(desc[f][c])
        
        grilla.append(fila)
    
    return grilla


def moverse(grilla, playerPos, goalsPos, boxesPos, direccion):
    aux_grilla = regenerate(grilla, playerPos, goalsPos, boxesPos)
    if puede_moverse(aux_grilla, playerPos, goalsPos, boxesPos, direccion):
        dx, dy = direccion
        playerPos = (playerPos[0] + dx, playerPos[1] + dy)
            
        # Mover bloque si el jugador empuja uno
        boxesPos = [(bx + dx, by + dy) if (bx, by) == playerPos else (bx, by) for bx, by in boxesPos]
        
    return playerPos, boxesPos  # Devolver posiciones sin cambios si el movimiento no es posible
    
def puede_moverse(grilla, playerPos, goalsPos, boxesPos, direccion):
    aux_grilla = regenerate(grilla, playerPos, goalsPos, boxesPos)
    dx, dy = direccion
    nueva_pos = (playerPos[0] + dx, playerPos[1] + dy)
        
    # No puede moverse si la nueva posición es una pared
    if aux_grilla[nueva_pos[0]][nueva_pos[1]] == PARED:
        return False
        
    # Si la nueva posición es un bloque, verificar si el bloque puede moverse
    if nueva_pos in boxesPos:
        nueva_pos_bloque = (nueva_pos[0] + dx, nueva_pos[1] + dy)
            
        # Verificar si el bloque se movería a una pared o a otro bloque
        if aux_grilla[nueva_pos_bloque[0]][nueva_pos_bloque[1]] in PARED or nueva_pos_bloque in boxesPos:
            return False
        
    return True


def regenerate(board, playerPos, goalsPos, boxesPos):
    board_copy = [row[:] for row in board]
    board_copy[playerPos[0]][playerPos[1]] = JUGADOR

    for i in goalsPos:
        board_copy[i[0]][i[1]] = OBJETIVO

    for i in boxesPos:
        board_copy[i[0]][i[1]] = CAJA

    return board_copy
