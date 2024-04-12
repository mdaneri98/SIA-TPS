
def avg_performance(population):
    return sum([character.performance() for character in population]) / len(population)

def check_content(population, previous_population, delta):
    pop_avg_performance = avg_performance(population)
    previous_pop_avg_performance = avg_performance(previous_population)

    return abs(pop_avg_performance - previous_pop_avg_performance) < delta

def check_optimal_fitness(population, optimal_fitness, optimal_fitness_error):
    best_character = max(population, key=lambda character: character.performance())
    best_character_performance =  best_character.performance()
    return (optimal_fitness - optimal_fitness_error) <= best_character_performance <= (optimal_fitness + optimal_fitness_error)

def check_max_generation(generation, max_generations): 
    if (generation >= max_generations):
        return True
 