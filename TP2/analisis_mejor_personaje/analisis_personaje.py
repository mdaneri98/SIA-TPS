import re
import matplotlib.pyplot as plt

def leer_datos(archivo):
    with open(archivo, 'r') as file:
        lines = file.readlines()
        datos = []
        for line in lines:
            # Utilizamos expresiones regulares para encontrar la primera palabra
            match_primera_palabra = re.search(r'^(\w+)', line)
            if match_primera_palabra:
                primera_palabra = match_primera_palabra.group(1)
            
            # Utilizamos expresiones regulares para encontrar el valor numérico de la performance
            match_performance = re.search(r'Performance:\s*([\d.]+)', line)
            if match_performance:
                performance = float(match_performance.group(1))
            
            # Guardamos la primera palabra y la performance en una tupla y la añadimos a la lista de datos
            datos.append((primera_palabra, performance))
    return datos

def graficar_diagrama_puntos(datos):
    # Extraer datos de selección y performance
    selecciones = [dato[0] for dato in datos]
    performances = [dato[1] for dato in datos]
    print(max(performances))

    # Crear el diagrama de puntos
    plt.figure(figsize=(8, 6))
    plt.scatter(selecciones, performances, color='blue')
    plt.xlabel('Selección')
    plt.ylabel('Fitness')
    plt.title('Selección vs Mejor fitness')
    plt.grid(True)
    plt.show()

archivo = "infiltrado_select.txt"
datos = leer_datos(archivo)
graficar_diagrama_puntos(datos)
