from typing import Dict, List
from character import Character
import random
from selection import *
from arguments import ProgramArguments

# Definir constantes
NUM_GENERACIONES = 3
TAMANO_POBLACION = 7
NUM_INDIVIDUOS_ELITE = 4
PROBABILIDAD_MUTACION = 0.1


def cruce_un_punto(first_parent: Character, second_parent: Character) -> tuple:
    # Obtener los genes de cada padre
    genes1 = first_parent.get_genes()
    genes2 = second_parent.get_genes()
    
    # Elegir un punto de cruce al azar dentro del rango más pequeño de items de ambos padres
    min_len = min(len(genes1), len(genes2))
    punto_cruce = random.randint(0, min_len - 1)

    # Efectuamos la cruza de genes.
    child_genes1 = genes2[:punto_cruce] + genes1[punto_cruce:]
    child_genes2 = genes1[:punto_cruce] + genes2[punto_cruce:]
    
    child1 = Character.from_genes(child_genes1)
    child2 = Character.from_genes(child_genes2)

    return child1, child2

def calcular_aptitudes(population):
    return [individual.performance() for individual in population]

class GeneticAlgorithmEngine:

    def __init__(self, arguments: ProgramArguments):
        self.generation = 0
        self.population: Dict[int:List] = {}
        self.arguments = arguments


    def generate_initial(self):
        # Inicializa la lista para la generación actual si aún no existe
        if self.generation not in self.population:
            self.population[self.generation] = []
            
        for i in range(TAMANO_POBLACION):
            ind = Character.create_random_character()
            self.population[self.generation].append(ind)


    def crossover(self):
        current_population = self.population[self.generation]
        while len(current_population) < TAMANO_POBLACION:
                first_parent, second_parent = random.choices(current_population, k=2)
                first_child, second_child = cruce_un_punto(first_parent, second_parent)
                first_child = self.mutate(first_child)
                second_child = self.mutate(second_child)
                self.population[self.generation].extend([first_child, second_child])

    def mutate(self, character: Character):
        return character


    #def select(self):
    def select(self, selection_method, param=None):
        current_population = self.population[self.generation]
        aptitudes = calcular_aptitudes(current_population)
        #elite = seleccion_elitista(current_population, aptitudes)

        if selection_method.__name__ == 'seleccion_boltzmann' is not None and param:
            method = selection_method(current_population, aptitudes, param)
        elif param is not None and selection_method.__name__ in ['seleccion_torneo_deterministico', 'seleccion_torneo_probabilistico']:
            method = selection_method(current_population, aptitudes, param)
        else:
            method = selection_method(current_population, aptitudes)


        # Insertamos la nueva generación.
        self.generation += 1
        self.population[self.generation] = method


    def start(self):
        self.generate_initial()

        for _ in range(NUM_GENERACIONES):
            self.select(seleccion_ranking)
            

            # Imprimimos resultados.
            print(f"Generación {self.generation}:\n")
            for ch in self.population[self.generation]:
                print(f"{ch}")

            # Obtener el individuo con la mejor performance
            ind_best_performance = max(self.population[self.generation], key=lambda individuo: individuo.performance())
            print(f"Mejor desempeño = {ind_best_performance}")

        print("Algoritmo genético completado.")

