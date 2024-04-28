import matplotlib.pyplot as plt
import csv
import numpy as np

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

def calcular_desvio_estandar(archivo_csv):
    generacion_actual = -1
    desvio_estandar = []
    rendimientos_generacion = []

    with open(archivo_csv, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            # Verificar si es una nueva generación
            if row[0] == '0':
                # Calcular el desvío estándar para la generación anterior
                if rendimientos_generacion:
                    desvio_estandar.append(np.std(rendimientos_generacion))
                # Reiniciar variables para la nueva generación
                generacion_actual += 1
                rendimientos_generacion = []
            # Agregar el rendimiento del personaje
            rendimientos_generacion.append(float(row[2]))  # Suponiendo que la columna del rendimiento es la tercera (índice 2)

        # Añadir el desvío estándar de la última generación
        if rendimientos_generacion:
            desvio_estandar.append(np.std(rendimientos_generacion))

    return desvio_estandar

# Función para graficar el promedio de rendimiento versus el número de generación
def graficar_promedio_performance(promedio_performance1, promedio_performance2, 
                                  promedio_performance3, promedio_performance4,
                                  desvio_performance1, desvio_performance2, 
                                  desvio_performance3, desvio_performance4,
                                  num_generaciones):

    # Generar los valores para el eje x (números enteros desde 0 hasta num_generaciones - 1)
    ejex = range(num_generaciones)
    plt.plot(ejex, promedio_performance1, linestyle='-', color='blue', label='Un punto')
    plt.plot(ejex, promedio_performance2, linestyle='-', color='red', label='Dos puntos')
    plt.plot(ejex, promedio_performance3, linestyle='-', color='green', label='Cruce anular')
    plt.plot(ejex, promedio_performance4, linestyle='-', color='orange', label='Cruce uniforme')

    # Agregar sombra de desvío estándar
    plt.fill_between(ejex, np.array(promedio_performance1) - np.array(desvio_performance1),
                     np.array(promedio_performance1) + np.array(desvio_performance1), color='blue', alpha=0.3)
    plt.fill_between(ejex, np.array(promedio_performance2) - np.array(desvio_performance2),
                     np.array(promedio_performance2) + np.array(desvio_performance2), color='red', alpha=0.3)
    plt.fill_between(ejex, np.array(promedio_performance3) - np.array(desvio_performance3),
                     np.array(promedio_performance3) + np.array(desvio_performance3), color='green', alpha=0.3)
    plt.fill_between(ejex, np.array(promedio_performance4) - np.array(desvio_performance4),
                    np.array(promedio_performance4) + np.array(desvio_performance4), color='orange', alpha=0.3)


    plt.xticks(range(0, num_generaciones, num_generaciones // 5))  # Establecer los valores del eje x
    plt.xlabel('Generación')
    plt.ylabel('Fitness')
    plt.title(f'Promedio de Fitness por Generación')
    plt.grid(True)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.4))  # Mostrar leyenda en el lado derecho
    plt.show()

archivo_1 = "datos_Arquero1.csv"
archivo_2 = "datos_Arquero2.csv"
archivo_3 = "datos_Arquero3.csv"
archivo_4 = "datos_Arquero4.csv"

promedio_performance1, num_generaciones1 = calcular_promedio_performance(archivo_1)
promedio_performance2, num_generaciones2 = calcular_promedio_performance(archivo_2)
promedio_performance3, num_generaciones3 = calcular_promedio_performance(archivo_3)
promedio_performance4, num_generaciones4 = calcular_promedio_performance(archivo_4)

desvio_performance1 = calcular_desvio_estandar(archivo_1)
desvio_performance2 = calcular_desvio_estandar(archivo_2)
desvio_performance3 = calcular_desvio_estandar(archivo_3)
desvio_performance4 = calcular_desvio_estandar(archivo_4)

num_generaciones = num_generaciones1

graficar_promedio_performance(promedio_performance1, promedio_performance2, 
                              promedio_performance3, promedio_performance4,
                              desvio_performance1, desvio_performance2, 
                              desvio_performance3, desvio_performance4,
                              num_generaciones)