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

# Función para calcular el desvío estándar por generación
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

# Función para graficar el promedio de rendimiento versus el número de generación con sombra de desvío estándar
def graficar_promedio_performance(promedio_performance1, promedio_performance2, 
                                  promedio_performance3, promedio_performance4, 
                                  promedio_performance5, promedio_performance6, 
                                  promedio_performance7, 
                                  num_generaciones,
                                  desvio_performance1, desvio_performance2, 
                                  desvio_performance3, desvio_performance4, 
                                  desvio_performance5, desvio_performance6, 
                                  desvio_performance7):
    # Generar los valores para el eje x (números enteros desde 0 hasta num_generaciones - 1)
    ejex = range(num_generaciones)
    plt.plot(ejex, promedio_performance1, linestyle='-', color='blue', label='Elitista')
    plt.plot(ejex, promedio_performance2, linestyle='-', color='red', label='Ranking')
    plt.plot(ejex, promedio_performance3, linestyle='-', color='green', label='Ruleta')
    plt.plot(ejex, promedio_performance4, linestyle='-', color='orange', label='Universal')
    plt.plot(ejex, promedio_performance5, linestyle='-', color='purple', label='Boltzmann')
    plt.plot(ejex, promedio_performance6, linestyle='-', color='cyan', label='Deterministico')
    plt.plot(ejex, promedio_performance7, linestyle='-', color='magenta', label='Probabilistico')

    # Agregar sombra de desvío estándar
    plt.fill_between(ejex, np.array(promedio_performance1) - np.array(desvio_performance1),
                     np.array(promedio_performance1) + np.array(desvio_performance1), color='blue', alpha=0.3)
    plt.fill_between(ejex, np.array(promedio_performance2) - np.array(desvio_performance2),
                     np.array(promedio_performance2) + np.array(desvio_performance2), color='red', alpha=0.3)
    plt.fill_between(ejex, np.array(promedio_performance3) - np.array(desvio_performance3),
                     np.array(promedio_performance3) + np.array(desvio_performance3), color='green', alpha=0.3)
    plt.fill_between(ejex, np.array(promedio_performance4) - np.array(desvio_performance4),
                     np.array(promedio_performance4) + np.array(desvio_performance4), color='orange', alpha=0.3)
    plt.fill_between(ejex, np.array(promedio_performance5) - np.array(desvio_performance5),
                     np.array(promedio_performance5) + np.array(desvio_performance5), color='purple', alpha=0.3)
    plt.fill_between(ejex, np.array(promedio_performance6) - np.array(desvio_performance6),
                     np.array(promedio_performance6) + np.array(desvio_performance6), color='cyan', alpha=0.3)
    plt.fill_between(ejex, np.array(promedio_performance7) - np.array(desvio_performance7),
                     np.array(promedio_performance7) + np.array(desvio_performance7), color='magenta', alpha=0.3)

    plt.xticks(range(0, num_generaciones, num_generaciones // 5))  # Establecer los valores del eje x
    plt.xlabel('Generación')
    plt.ylabel('Fitness')
    plt.title(f'Promedio de Fitness por Generación - Defensor')
    plt.grid(True)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.4))  # Mostrar leyenda en el lado derecho
    plt.show()

# Calcular promedio de rendimiento por generación y cantidad de generaciones para cada archivo
archivo_1 = "datos_Defensor1.csv"
archivo_2 = "datos_Defensor2.csv"
archivo_3 = "datos_Defensor3.csv"
archivo_4 = "datos_Defensor4.csv"
archivo_5 = "datos_Defensor5.csv"
archivo_6 = "datos_Defensor6.csv"
archivo_7 = "datos_Defensor7.csv"

promedio_performance1, num_generaciones1 = calcular_promedio_performance(archivo_1)
promedio_performance2, num_generaciones2 = calcular_promedio_performance(archivo_2)
promedio_performance3, num_generaciones3 = calcular_promedio_performance(archivo_3)
promedio_performance4, num_generaciones4 = calcular_promedio_performance(archivo_4)
promedio_performance5, num_generaciones5 = calcular_promedio_performance(archivo_5)
promedio_performance6, num_generaciones6 = calcular_promedio_performance(archivo_6)
promedio_performance7, num_generaciones7 = calcular_promedio_performance(archivo_7)

desvio_performance1 = calcular_desvio_estandar(archivo_1)
desvio_performance2 = calcular_desvio_estandar(archivo_2)
desvio_performance3 = calcular_desvio_estandar(archivo_3)
desvio_performance4 = calcular_desvio_estandar(archivo_4)
desvio_performance5 = calcular_desvio_estandar(archivo_5)
desvio_performance6 = calcular_desvio_estandar(archivo_6)
desvio_performance7 = calcular_desvio_estandar(archivo_7)

# Tomar el máximo número de generaciones entre todos los archivos
num_generaciones = max(num_generaciones1, num_generaciones2, num_generaciones3,
                       num_generaciones4, num_generaciones5, num_generaciones6, num_generaciones7)

# Graficar promedio de rendimiento versus número de generación para cada archivo con sombra de desvío estándar
graficar_promedio_performance(promedio_performance1, promedio_performance2, 
                              promedio_performance3, promedio_performance4, 
                              promedio_performance5, promedio_performance6, 
                              promedio_performance7, 
                              num_generaciones,
                              desvio_performance1, desvio_performance2, 
                              desvio_performance3, desvio_performance4, 
                              desvio_performance5, desvio_performance6, 
                              desvio_performance7)