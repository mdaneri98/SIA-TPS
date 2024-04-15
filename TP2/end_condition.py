import time
import math


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
 
def standard_deviation(dataset):
    mean = sum(dataset) / len(dataset)
    variance = sum([(x - mean) ** 2 for x in dataset]) / len(dataset)
    return math.sqrt(variance)

def calculate_diversity(population):
    population_genes = [[], [], [], [], [], []]
    
    for character in population:
        genes = character.get_genes()
        population_genes[0].append(genes["strength"])
        population_genes[1].append(genes["agility"])
        population_genes[2].append(genes["expertise"])
        population_genes[3].append(genes["resistance"])
        population_genes[4].append(genes["life"])
        population_genes[5].append(character.height())
    
    standard_deviations = [standard_deviation(stat) for stat in population_genes]
    return sum(standard_deviations) / len(standard_deviations)

#Diversidad insuficiente: Si la población de soluciones generadas por el algoritmo se vuelve 
#muy homogénea o pierde diversidad y el algoritmo podría quedarse atrapado en un óptimo local.
def check_structural_end_condition(population, previous_population, delta):
    population_diversity = calculate_diversity(population)
    previous_population_diversity = calculate_diversity(previous_population)
    return abs(population_diversity - previous_population_diversity) < delta
