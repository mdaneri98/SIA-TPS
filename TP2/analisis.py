import matplotlib.pyplot as plt
import csv
import matplotlib.pyplot as plt


# Función para leer los datos del archivo CSV y calcular el promedio de rendimiento por generación
def calcular_promedio_performance(archivo_csv):
    generacion_actual = -1
    performance_generacion = []
    promedio_performance = []
    total_performance = 0
    cantidad_personajes = 0

    with open(archivo_csv, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            # Verificar si es una nueva generación
            if row[0] == '0':
                # Calcular el promedio de rendimiento para la generación anterior
                if cantidad_personajes > 0:
                    promedio_performance.append(total_performance / cantidad_personajes)
                    performance_generacion.append(generacion_actual)
                # Reiniciar variables para la nueva generación
                generacion_actual += 1
                total_performance = 0
                cantidad_personajes = 0
            # Sumar el rendimiento del personaje a total_performance
            total_performance += float(row[2])  # Suponiendo que la columna del rendimiento es la tercera (índice 2)
            cantidad_personajes += 1

        # Añadir el promedio de la última generación
        if cantidad_personajes > 0:
            promedio_performance.append(total_performance / cantidad_personajes)
            performance_generacion.append(generacion_actual)

    return promedio_performance, generacion_actual + 1

# Función para graficar el promedio de rendimiento versus el número de generación
def graficar_promedio_performance(promedio_performance1, promedio_performance2, num_generaciones):
    # Generar los valores para el eje x (números enteros desde 0 hasta num_generaciones - 1)
    ejex = range(num_generaciones)
    plt.plot(ejex, promedio_performance1, linestyle='-')
    plt.plot(ejex, promedio_performance2, linestyle='-')
    plt.xticks(ejex)  # Establecer los valores del eje x
    plt.xlabel('Generación')
    plt.ylabel('Promedio de Performance')
    plt.title(f'Promedio de Performance por Generación')
    plt.grid(True)


    plt.show()

# Calcular promedio de rendimiento por generación y cantidad de generaciones
#promedio_performance, num_generaciones = calcular_promedio_performance(archivo_csv)

# Graficar promedio de rendimiento versus número de generación
#graficar_promedio_performance(promedio_performance, num_generaciones)
archivo_csv1 = "datos_Arquero1.csv"
archivo_csv2 = "datos_Arquero2.csv"

# Calcular promedio de rendimiento por generación y cantidad de generaciones para ambos conjuntos de datos
promedio_performance1, num_generaciones1 = calcular_promedio_performance(archivo_csv1)
promedio_performance2, num_generaciones2 = calcular_promedio_performance(archivo_csv2)

# Tomar el máximo número de generaciones entre ambos conjuntos de datos
num_generaciones = max(num_generaciones1, num_generaciones2)

# Graficar promedio de rendimiento versus número de generación para ambos conjuntos de datos
graficar_promedio_performance(promedio_performance1, promedio_performance2, num_generaciones)