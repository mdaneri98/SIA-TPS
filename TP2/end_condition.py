import time
import math
from character import Character, Character_type

def avg_performance(population):
    return sum([character.performance() for character in population]) / len(population)

def check_content(population, previous_population, delta,start_t,time_limit):
    pop_avg_performance = avg_performance(population)
    previous_pop_avg_performance = avg_performance(previous_population)

    return abs(pop_avg_performance - previous_pop_avg_performance) < delta

def check_optimal_fitness(population, optimal_fitness, optimal_fitness_error,start_t,time_limit):
    best_character = max(population, key=lambda character: character.performance())
    best_character_performance =  best_character.performance()
    return (optimal_fitness - optimal_fitness_error) <= best_character_performance <= (optimal_fitness + optimal_fitness_error)

def check_max_generation(generation, max_generations): 
    if (generation >= max_generations):
        return True

def save_genes(population):
    population_genes = [[], [], [], [], []]

    for character in population:
        genes = character.get_genes()
        population_genes[0].append(genes[2])
        population_genes[1].append(genes[3])
        population_genes[2].append(genes[4])
        population_genes[3].append(genes[5])
        population_genes[4].append(genes[6])

    return population_genes


#Diversidad insuficiente: Si la población de soluciones generadas por el algoritmo se vuelve 
#muy homogénea o pierde diversidad y el algoritmo podría quedarse atrapado en un óptimo local.
def check_structural(population, previous_population, delta,start_t,time_limit): 
    if (time.time()-start_t >= time_limit):
        return True
    diversidad = 0
    population_genes = save_genes(population)
    prev_population_genes = save_genes(previous_population)

    for i in range(len(population_genes)):
        for j in range(len(population_genes[i])):
            if abs(population_genes[i][j] - prev_population_genes[i][j]) > delta:
                diversidad += 1

    return diversidad >= delta

