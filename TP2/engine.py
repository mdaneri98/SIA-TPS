from typing import Dict, List
from character import Character
import random
import time
from selection import *
from mutacion import *
from arguments import ProgramArguments
from crossover import *
from end_condition import *
import csv


def calcular_aptitudes(population):
    return [individual.performance() for individual in population]

def get_type(arguments) -> CharacterType:
    type_str = arguments['poblacion']['tipo']
    if type_str == 'guerrero':
        return CharacterType.Guerrero
    elif type_str == 'arquero':
        return CharacterType.Arquero
    elif type_str == 'infiltrado':
        return CharacterType.Infiltrado
    elif type_str == 'defensor':
        return CharacterType.Defensor
    
    return "No valid character type"


class GeneticAlgorithmEngine:

    def __init__(self, arguments):
        self.generation = 0
        self.population: Dict[int:List] = {}
        self.arguments = arguments


    def add_generation(self, new_population: List):
        self.generation += 1
        self.population[self.generation] = list(new_population)


    def generate_initial(self, n, characterType: CharacterType):
        # Inicializa la lista para la generación actual si aún no existe
        if self.generation not in self.population:
            self.population[self.generation] = []
        
        for _ in range(n):
            ind = Character.create_random_character(characterType)
            self.population[self.generation].append(ind)


    def crossover(self, crossover_method_name: str, k: int, probability: float):
        current_population = self.population[self.generation]
        
        childs = []
        # Por cada dos padres, dos hijos.
        for _ in range(0, k, 2):
                first_parent, second_parent = random.choices(current_population, k=2)

                if crossover_method_name == 'cruce_un_punto':
                    method = Crossover.cruce_un_punto
                    args = (first_parent, second_parent)
                elif crossover_method_name == 'cruce_dos_puntos':
                    method = Crossover.cruce_dos_puntos
                    args = (first_parent, second_parent)
                elif crossover_method_name == 'cruce_anular':
                    method = Crossover.cruce_anular
                    args = (first_parent, second_parent)
                elif crossover_method_name == 'cruce_uniforme':
                    method = Crossover.cruce_uniforme
                    args = (first_parent, second_parent, probability)

                first_child, second_child = method(*args)
                childs.append(first_child)
                
                # Si k es impar -> no hay que añadir el segundo hijo en el último step.
                if len(childs) == k:
                    break
                childs.append(second_child)
        
        return childs


    # def mutate(self, character: Character):
    #     return character
    
    def mutate(self, selection_method_name: str, individuo: Character, delta_items : float,delta_height : float, proba_mutacion : float, generacion : int):
        if selection_method_name == 'mutacion_multigen_uniform':
            character = mutacion_multigen(individuo, generacion , proba_mutacion, delta_items, delta_height, True)
        if selection_method_name == 'mutacion_multigen_no_uniforme':
             character = mutacion_multigen(individuo, generacion , proba_mutacion, delta_items, delta_height, False)
        if selection_method_name == 'mutacion_gen_uniform':
            character = mutacion_gen(individuo, generacion , proba_mutacion, delta_height ,True)
        if selection_method_name == 'mutacion_gen_no_uniforme' :
            character = mutacion_gen(individuo, generacion , proba_mutacion, delta_height ,False)
        return character


    def select(self, selection_method_name: str, population: List, n: int, k: int, m: int, threshold: float, temperatura_inicial: float):
        aptitudes = calcular_aptitudes(population)
        if selection_method_name == 'seleccion_boltzmann':
            selected_population = seleccion_boltzmann(population, aptitudes, k, temperatura_inicial)
        elif selection_method_name == 'seleccion_ruleta':
            selected_population = seleccion_ruleta(population, aptitudes, k)
        elif selection_method_name == 'seleccion_ranking':
            selected_population = seleccion_ranking(population, aptitudes, k)
        elif selection_method_name == 'seleccion_universal':
            selected_population = seleccion_universal(population, aptitudes, k)
        elif selection_method_name == 'seleccion_torneo_deterministico':
            selected_population = seleccion_torneo_deterministico(population, aptitudes, k, m)
        elif selection_method_name == 'seleccion_torneo_probabilistico':
            selected_population = seleccion_torneo_probabilistico(population, aptitudes, k, threshold)
        else:
            selected_population = seleccion_elitista(population, aptitudes, k)

        # Retornamos la nueva generación.
        return selected_population


    archivo_stop = 'archivo_stop.csv'
    def stop(self, stop_method, max_generations, optimal_fitness, optimal_fitness_error, delta,start_t,time_limit,gen,deltaE):
        if stop_method == 'cantidad_generaciones':
            with open('archivo_stop.csv', 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                if check_optimal_fitness(self.population[self.generation], optimal_fitness, optimal_fitness_error, start_t, time_limit):
                    writer.writerow([gen, optimal_fitness, 'optimal'])
                if check_content(self.population[self.generation], self.population[self.generation-1], delta,start_t,time_limit):
                    writer.writerow([gen, optimal_fitness, 'content'])
                if check_structural(self.population[self.generation], self.population[self.generation-1], deltaE,start_t,time_limit):
                    writer.writerow([gen, optimal_fitness, 'estructural'])
            
            return (check_max_generation(self.generation , max_generations))
        elif stop_method == 'solucion_aceptable':
            return (check_optimal_fitness(self.population[self.generation], optimal_fitness, optimal_fitness_error,start_t,time_limit))
        elif stop_method == 'contenido':
            return (check_content(self.population[self.generation], self.population[self.generation-1], delta,start_t,time_limit))
        elif stop_method == 'estructural':
            return (check_structural(self.population[self.generation], self.population[self.generation-1], deltaE,start_t,time_limit))
        
        return "No stop method given"


    def start(self):

        # Poblacion 
        n = int(self.arguments['poblacion']['cantidad_poblacion'])
        k = int(self.arguments['poblacion']['k'])
        characterType = get_type(self.arguments)
        #CSV
        archivo_csv = f'datos_{characterType.name}.csv'

        # Seleccion
        selection_method_name1 = self.arguments['seleccion']['metodo1']
        selection_method_name2 = self.arguments['seleccion']['metodo2']
        selection_method_name3 = self.arguments['seleccion']['metodo3']
        selection_method_name4 = self.arguments['seleccion']['metodo4']
        A = float(self.arguments['seleccion']['a'])
        B = float(self.arguments['seleccion']['b'])
        m = int(self.arguments['seleccion']['m'])
        threshold = float(self.arguments['seleccion']['threshold'])
        temperatura_inicial = float(self.arguments['seleccion']['temperatura_inicial'])
        
        # Crossover
        crossover_method_name = self.arguments['crossover']['metodo']
        crossover_probability = float(self.arguments['crossover']['probability'])

        # Condicion de corte
        start_time = time.time()
        time_limit = float(self.arguments['corte']['time_limit'])
        stop_method = self.arguments['corte']['metodo']
        max_generations = int(self.arguments['corte']['max_generaciones'])
        optimal_fitness = int(self.arguments['corte']['optimal_fitness'])
        optimal_fitness_error = int(self.arguments['corte']['optimal_fitness_error'])
        delta = float(self.arguments['corte']['delta1'])
        deltaE = int(self.arguments['corte']['delta2'])

        #Mutacion
        selection_mut_name1 = self.arguments['mutacion']['metodo1']
        selection_mut_name2 = self.arguments['mutacion']['metodo2']
        selection_mut_name3 = self.arguments['mutacion']['metodo3']
        selection_mut_name4 = self.arguments['mutacion']['metodo4']
        probabilidad_mutacion = float(self.arguments['mutacion']['probabilidad_mutacion'])
        delta_items = float(self.arguments['mutacion']['delta_items'])
        delta_height =  float(self.arguments['mutacion']['delta_height'])
        
        self.generate_initial(n, characterType)
        #print(f"Generación {self.generation}:\n")
        #for i, ch in enumerate(self.population[self.generation]):
        #    print(f"{i}: {ch}")

        gen = 0
        stop = False
        while not stop:
            # --- Generamos la nueva población --- 
            # Seleccionamos los k padres, que harán el 'crossover' y generarán el nuevo conj. de hijos.
            count_method1 = ceil(A*k)
            count_method2 = k - count_method1

            current_population = self.population[self.generation]
            selection1 = self.select(selection_method_name1, current_population, n, count_method1, m, threshold, temperatura_inicial)
            selection2 = self.select(selection_method_name2, current_population, n, count_method2, m, threshold, temperatura_inicial)
            parents_population = selection1 + selection2

            # --- Realizamos el crossover ---
            # Generamos la cruza de los K padres, generando K hijos.
            childs_population = self.crossover(crossover_method_name, k, crossover_probability)

            # --- Realizamos la mutación ---           
            for i,child in enumerate(childs_population):
                childs_population[i] = self.mutate(selection_mut_name1, child, delta_items, delta_height, probabilidad_mutacion, self.generation)
                

            # --- Reemplazamos ---
            count_method3 = ceil(B*n)
            count_method4 = n - count_method3

            # Seleccionamos N + K individuos de la nueva población 
            big_population = current_population + childs_population

            selection3 = self.select(selection_method_name3, big_population, n + k, count_method3, m, threshold, temperatura_inicial)
            selection4 = self.select(selection_method_name4, big_population, n + k, count_method4, m, threshold, temperatura_inicial)

            new_population = selection3 + selection4
            self.add_generation(new_population)

            # Imprimimos la nueva generación.
            #print(f"Generación {self.generation}:\n")
            for i, ch in enumerate(self.population[self.generation]):
            #    print(f"{i}: {ch}")
                with open(archivo_csv, 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow([i, ch.ch_type, ch.performance(), ch.attack, ch.defense, ch.itemPoints])

            # Obtener el individuo con la mejor performance
            ind_best_performance = (max(self.population[self.generation], key=lambda individuo: individuo.performance()), gen)
            

            # Chequeamos la stop condition. No varía, pero si elejimos 'content' usa la población anterior y rompería.
            stop = self.stop(stop_method, max_generations, optimal_fitness, optimal_fitness_error, delta,start_time,time_limit, gen, deltaE)
            gen+=1
            #print(f"args {stop_method_args}")
            #stop_method_f(stop_method_args)

        # 2 print para saber si hay bien mutaccion entre el inicio y la fin 

        print(ind_best_performance[0], ind_best_performance[1])
        #print(max(self.population[0],key=lambda individuo: individuo.performance()))
        #print(max(self.population[self.generation],key=lambda individuo: individuo.performance()))
            



