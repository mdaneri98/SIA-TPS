import random
import math

# Definir constantes
NUM_GENERACIONES = 100
TAMANO_POBLACION = 100
NUM_INDIVIDUOS_ELITE = 10
PROBABILIDAD_MUTACION = 0.1

def seleccion_elitista(poblacion, aptitudes, k):
    # Ordenar los índices de la población en función de la aptitud de cada individuo
    elite_indices = sorted(range(len(aptitudes)), key=lambda i: aptitudes[i], reverse=True)[:k]
    # Seleccionar a los individuos elite según los índices obtenidos
    elite = [poblacion[i] for i in elite_indices]
    return elite


def seleccion_ruleta(poblacion, aptitudes, k):
    suma_aptitudes = sum(aptitudes)
    aptitud_relativa = [aptitud / suma_aptitudes for aptitud in aptitudes]
    seleccionados = []
    for _ in range(k):
        r = random.random()
        acumulada = 0
        for i, probabilidad in enumerate(aptitud_relativa):
            acumulada += probabilidad
            if r <= acumulada:
                seleccionados.append(poblacion[i])
                break
    return seleccionados


def seleccion_universal(poblacion, aptitudes, k):
    # Calcula la suma total de las aptitudes
    suma_aptitudes = sum(aptitudes)

    # Generamos la lista de aptitudes relativas acumuladas qi
    aptitudes_relativas_acumuladas = []
    acumulado = 0
    for aptitud in aptitudes:
        acumulado += aptitud / suma_aptitudes
        aptitudes_relativas_acumuladas.append(acumulado)

    # Genera números aleatorios y selecciona k individuos
    seleccionados = []
    for _ in range(k):
        r = random.random()
        for i, probabilidad in enumerate(aptitudes_relativas_acumuladas):
            if r <= probabilidad:
                seleccionados.append(poblacion[i])
                break
    return seleccionados


def seleccion_boltzmann(poblacion, aptitudes, k, temperatura):
    probabilidades = [math.exp(aptitud / temperatura) for aptitud in aptitudes]
    suma_probabilidades = sum(probabilidades)
    # Normalizamos las probabilidades para asegurarnos de que sumen 1 y puedan utilizarse como probabilidades de selección.
    probabilidades_normalizadas = [prob / suma_probabilidades for prob in probabilidades]

    seleccionados = []
    for _ in range(k):
        r = random.random()
        acumulada = 0
        for i, probabilidad in enumerate(probabilidades_normalizadas):
            acumulada += probabilidad
            if r <= acumulada:
                seleccionados.append(poblacion[i])
                break

    return seleccionados


def seleccion_ranking(poblacion, aptitudes, k):
    # Ordenar la población y las aptitudes en función de las aptitudes de los individuos
    # Ordena la población basada únicamente en las aptitudes, sin intentar desempatar con las instancias de Character directamente
    poblacion_ordenada = [individuo for aptitud, individuo in sorted(zip(aptitudes, poblacion), key=lambda x: x[0], reverse=True)]
    
    # Calcular las probabilidades de selección basadas en el ranking
    n = len(poblacion)
    probabilidades = [(2 * (n - i)) / (n * (n + 1)) for i in range(n)]
    
    # Seleccionar individuos utilizando las probabilidades de selección
    seleccionados = []
    for _ in range(k):
        r = random.random()
        acumulada = 0
        for i, probabilidad in enumerate(probabilidades):
            acumulada += probabilidad
            if r <= acumulada:
                seleccionados.append(poblacion_ordenada[i])
                break
    
    return seleccionados


def seleccion_torneo_deterministico(poblacion, aptitudes, k, m):
    '''
        @param k: Cantidad de individuos a seleccionar.
        @param m: Cantidad de individuos por torneo.
    '''
    seleccionados = []
    for j in range(k):
        # Seleccionamos M individuos de la población de tamaño N.
        torneo = random.sample(range(len(poblacion)), m)  

        # Seleccionar el individuo con la mayor aptitud en el torneo
        ganador_torneo = max(torneo, key=lambda i: aptitudes[i])
        seleccionados.append(poblacion[ganador_torneo])

    return seleccionados


def seleccion_torneo_probabilistico(poblacion, aptitudes, k, threshold):
    seleccionados = []
    for _ in range(k):
        # Seleccionamos 2 individuos de la población de tamaño N.
        torneo = random.sample(range(len(poblacion)), 2)
        
        r = random.random()
        if r < threshold:
            # Seleccionar el individuo con la mayor aptitud en el torneo
            ganador_torneo = max(torneo, key=lambda i: aptitudes[i])
        else:
            ganador_torneo = min(torneo, key=lambda i: aptitudes[i])
        seleccionados.append(poblacion[ganador_torneo])
        
    return seleccionados
