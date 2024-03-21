import arcade

# Definir el tamaño de la ventana y el tamaño de la celda
ANCHO = 550
ALTO = 550
TAMANO_CELDA = 30

nombre_archivo = "C:\\Users\\desir\\VSCProjects\\SIA-TPS\\TP1\\boards\\level1.txt"

# Lista para almacenar las líneas del tablero
tablero = []

# Abrir el archivo y leer cada línea
with open(nombre_archivo, "r") as archivo:
    for linea in archivo:
        # Eliminar el carácter de salto de línea al final de cada línea y agregarla al tablero
        tablero.append(linea.strip())

# Imprimir el tablero para verificar
for fila in tablero:
    print(fila)


def dibujar_tablero():
    for fila in range(len(tablero)):
        for columna in range(len(tablero[fila])):
            caracter = tablero[fila][columna]
            color = arcade.color.BLACK
            if caracter == "#":
                color = arcade.color.BLACK
            elif caracter == ".":
                color = arcade.color.GREEN
            elif caracter == " ":
                color = arcade.color.WHITE
            elif caracter == "-":
                color = arcade.color.WHITE
            elif caracter == "@":
                color = arcade.color.BLUE
            elif caracter == "$":
                color = arcade.color.GO_GREEN

            arcade.draw_rectangle_filled(
                columna * TAMANO_CELDA + TAMANO_CELDA / 2,
                (len(tablero) - fila - 1) * TAMANO_CELDA + TAMANO_CELDA / 2,
                TAMANO_CELDA,
                TAMANO_CELDA,
                color
            )

def on_draw(delta_time):
    arcade.start_render()
    dibujar_tablero()

def main():
    arcade.open_window(ANCHO, ALTO, "Tablero Sokoban")
    arcade.set_background_color(arcade.color.WHITE)

    arcade.schedule(on_draw, 1 / 60)

    arcade.run()

if __name__ == "__main__":
    main()
