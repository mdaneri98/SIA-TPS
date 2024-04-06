import random
import math

# Definir constantes
NUM_GENERACIONES = 100
TAMANO_POBLACION = 100
NUM_INDIVIDUOS_ELITE = 10
PROBABILIDAD_MUTACION = 0.1

def seleccion_elitista(poblacion, aptitudes):
    # Ordenar los índices de la población en función de la aptitud de cada individuo
    elite_indices = sorted(range(len(aptitudes)), key=lambda i: aptitudes[i], reverse=True)[:NUM_INDIVIDUOS_ELITE]
    # Seleccionar a los individuos elite según los índices obtenidos
    elite = [poblacion[i] for i in elite_indices]
    return elite


def seleccion_ruleta(poblacion, aptitudes):
    suma_aptitudes = sum(aptitudes)
    probabilidad_seleccion = [aptitud / suma_aptitudes for aptitud in aptitudes]
    seleccionados = []
    for _ in range(len(poblacion)):
        r = random.random()
        acumulada = 0
        for i, probabilidad in enumerate(probabilidad_seleccion):
            acumulada += probabilidad
            if r <= acumulada:
                seleccionados.append(poblacion[i])
                break
    return seleccionados

def seleccion_universal(poblacion, aptitudes):
    # Calcula la suma total de las aptitudes
    suma_aptitudes = sum(aptitudes)

    # Genera una lista acumulativa de probabilidades
    probabilidades_acumulativas = []
    acumulado = 0
    for aptitud in aptitudes:
        acumulado += aptitud / suma_aptitudes
        probabilidades_acumulativas.append(acumulado)

    # Genera números aleatorios y selecciona individuos
    seleccionados = []
    for _ in range(len(poblacion)):
        r = random.random()
        for i, probabilidad in enumerate(probabilidades_acumulativas):
            if r <= probabilidad:
                seleccionados.append(poblacion[i])
                break

    return seleccionados

def seleccion_boltzmann(poblacion, aptitudes, temperatura):
    probabilidades = [math.exp(aptitud / temperatura) for aptitud in aptitudes]
    suma_probabilidades = sum(probabilidades)
    # Normalizamos las probabilidades para asegurarnos de que sumen 1 y puedan utilizarse como probabilidades de selección.
    probabilidades_normalizadas = [prob / suma_probabilidades for prob in probabilidades]

    seleccionados = []
    for _ in range(len(poblacion)):
        r = random.random()
        acumulada = 0
        for i, probabilidad in enumerate(probabilidades_normalizadas):
            acumulada += probabilidad
            if r <= acumulada:
                seleccionados.append(poblacion[i])
                break
    return seleccionados

def seleccion_ranking(poblacion, aptitudes):
    # Ordenar la población y las aptitudes en función de las aptitudes de los individuos
    # Los individuos más aptos van a tener una mayor probabilidad de ser seleccionados
    poblacion_ordenada = [individuo for _, individuo in sorted(zip(aptitudes, poblacion), reverse=True)]
    aptitudes_ordenadas = sorted(aptitudes, reverse=True)
    
    # Calcular las probabilidades de selección basadas en el ranking
    n = len(poblacion)
    probabilidades = [(2 * (n - i)) / (n * (n + 1)) for i in range(n)]
    
    # Seleccionar individuos utilizando las probabilidades de selección
    seleccionados = []
    for _ in range(n):
        r = random.random()
        acumulada = 0
        for i, probabilidad in enumerate(probabilidades):
            acumulada += probabilidad
            if r <= acumulada:
                seleccionados.append(poblacion_ordenada[i])
                break
    
    return seleccionados

def seleccion_torneo_deterministico(poblacion, aptitudes, threshold):
    seleccionados = []
    for _ in range(len(poblacion)):
        torneo = random.sample(range(len(poblacion)), threshold)  # Seleccionar aleatoriamente el tamaño del torneo
        ganador_torneo = max(torneo, key=lambda i: aptitudes[i])  # Seleccionar el individuo con la mayor aptitud en el torneo
        seleccionados.append(poblacion[ganador_torneo])
    return seleccionados

def seleccion_torneo_probabilistico(poblacion, aptitudes, threshold):
    seleccionados = []
    for _ in range(len(poblacion)):
        torneo = random.sample(range(len(poblacion)), threshold)  # Seleccionar aleatoriamente el tamaño del torneo
        probabilidad_seleccion = [aptitudes[i] / sum(aptitudes) for i in torneo]  # Calcular la probabilidad de selección de cada individuo en el torneo
        ganador_torneo = random.choices(torneo, weights=probabilidad_seleccion)[0]  # Seleccionar un individuo del torneo con base en las probabilidades
        seleccionados.append(poblacion[ganador_torneo])
    return seleccionados
