import random
import math

# Definir constantes
NUM_GENERACIONES = 100
TAMANO_POBLACION = 100
NUM_INDIVIDUOS_ELITE = 10
PROBABILIDAD_MUTACION = 0.1

# Definir características de los items
CARACTERISTICAS_ITEMS = ["Fuerza", "Agilidad", "Pericia", "Resistencia", "Vida"]
PUNTOS_TOTALES = 150

# Definir clases de personajes y sus desempeños
CLASES_PERSONAJES = {
    "Guerrero": {"ataque": 0.6, "defensa": 0.4},
    "Arquero": {"ataque": 0.9, "defensa": 0.1},
    "Defensor": {"ataque": 0.1, "defensa": 0.9},
    "Infiltrado": {"ataque": 0.8, "defensa": 0.3}
}


# Definir funciones de utilidad
def calcular_desempeno(clase, atributos):
    ataque = (atributos["Agilidad"] + atributos["Pericia"]) * atributos["Fuerza"] * calcular_modificador_altura(
        atributos["Altura"])
    defensa = (atributos["Resistencia"] + atributos["Pericia"]) * atributos["Vida"] * calcular_modificador_altura(
        atributos["Altura"])
    return CLASES_PERSONAJES[clase]["ataque"] * ataque + CLASES_PERSONAJES[clase]["defensa"] * defensa


def calcular_modificador_altura(altura):
    if 1.3 <= altura <= 2.0:
        return 0.5 - (3 * altura - 5) ** 4 + (3 * altura - 5) ** 2 + altura / 2
    else:
        return 0


def generar_individuo():
    individuo = {}
    for caracteristica in CARACTERISTICAS_ITEMS:
        individuo[caracteristica] = random.uniform(0, PUNTOS_TOTALES)
    individuo['Altura'] = random.uniform(1.3, 2.0)  # Altura aleatoria entre 1.3 y 2.0 metros
    return individuo


def cruce_un_punto(padre1, padre2):
    punto_cruce = random.randint(1, len(padre1) - 1)
    hijo1 = {k: padre1[k] if i <= punto_cruce else padre2[k] for i, k in enumerate(padre1)}
    hijo2 = {k: padre2[k] if i <= punto_cruce else padre1[k] for i, k in enumerate(padre1)}
    return hijo1, hijo2


def mutacion_uniforme(individuo):
    for caracteristica in individuo:
        if random.random() < PROBABILIDAD_MUTACION:
            individuo[caracteristica] += random.uniform(-10, 10)
            individuo[caracteristica] = max(0, min(individuo[caracteristica], PUNTOS_TOTALES))
    return individuo

def mutacion_


def seleccion_elitista(poblacion, aptitudes):
    elite_indices = sorted(range(len(aptitudes)), key=lambda i: aptitudes[i], reverse=True)[:NUM_INDIVIDUOS_ELITE]
    elite = [poblacion[i] for i in elite_indices]
    return elite


def seleccion_ruleta(poblacion, aptitudes):
    suma_aptitudes = sum(aptitudes)
    probabilidad_seleccion = [aptitud / suma_aptitudes for aptitud in aptitudes]
    seleccionados = []
    for _ in range(TAMANO_POBLACION):
        r = random.random()
        acumulada = 0
        for i, probabilidad in enumerate(probabilidad_seleccion):
            acumulada += probabilidad
            if r <= acumulada:
                seleccionados.append(poblacion[i])
                break
    return seleccionados


def reemplazo_generacional(poblacion, nueva_generacion):
    return nueva_generacion


# Función principal del algoritmo genético
def algoritmo_genetico():
    poblacion = [generar_individuo() for _ in range(TAMANO_POBLACION)]

    for generacion in range(NUM_GENERACIONES):
        aptitudes = [calcular_desempeno("Guerrero", individuo) for individuo in poblacion]
        elite = seleccion_elitista(poblacion, aptitudes)

        nueva_generacion = elite[:]

        while len(nueva_generacion) < TAMANO_POBLACION:
            padre1, padre2 = random.choices(poblacion, k=2)
            hijo1, hijo2 = cruce_un_punto(padre1, padre2)
            hijo1 = mutacion_uniforme(hijo1)
            hijo2 = mutacion_uniforme(hijo2)
            nueva_generacion.extend([hijo1, hijo2])

        poblacion = reemplazo_generacional(poblacion, nueva_generacion)

        print(f"Generación {generacion + 1}: Mejor desempeño = {max(aptitudes)}")

    print("Algoritmo genético completado.")


# Ejecutar algoritmo genético
algoritmo_genetico()