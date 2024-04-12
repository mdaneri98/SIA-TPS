import random
from character import Character  # Importer la classe Character depuis le module character
from enum import Enum


# Définir les constantes
NUM_GENERACIONES = 100
TAMANO_POBLACION = 100
NUM_INDIVIDUOS_ELITE = 10
PROBABILIDAD_MUTACION = 0.1
PUNTOS_TOTALES = 150
DELTA_ITEMS = 10
DELTA_HEIGHT = 0.1
CharacterType = Enum('Type', ['Guerrero', 'Arquero', 'Defensor', 'Infiltrado'])


def calculate_probability_mutation(generation,uniform, probabilidad_mutacion)->float:
    if (uniform) : 
        return probabilidad_mutacion
    else :
        return (probabilidad_mutacion / (generation + 1))


def mutacion_multigen(individuo: Character, generation : int, probabilidad_mutacion : float, delta_items : float, delta_height : float, uniform =True):
    
    prob_mutacion = calculate_probability_mutation(generation,uniform, probabilidad_mutacion)

    genes = individuo.get_genes()
    new_type = genes[0]  # Initialiser avec la valeur actuelle
    new_height = genes[1]
    indices_mutations = []
    
    # Mutate the type of character (gene 0)
    # if random.random() < prob_mutacion:
        
    #     random_type = random.choice(list(CharacterType))
    #     new_type = random_type.name
    
    # Mutate the height (gene 1)
    if random.random() < prob_mutacion:
        
        new_height = genes[1] + random.uniform(-delta_height, delta_height)
        new_height = max(1.3, min(new_height, 2.0))  # Limit height between 1.3 and 2.0 meters
    

    # Mutate the rest of the genes (items)
    # we have to be careful 
    
    sum = 0
    
    for i in range(2, len(genes)):
        if random.random() < prob_mutacion:
            
            indices_mutations.append(i)
            sum += genes[i]
            

    nb_mutated = len(indices_mutations)

    if nb_mutated > 1 : 
        
        valeurs_originales = [genes[i] for i in indices_mutations]
        goal = -1
        while(goal!=sum):
            for i, valeur in zip(indices_mutations, valeurs_originales):
                genes[i] = valeur
            goal =0
            cst = round(random.uniform(-delta_items, delta_items),4)
            cst2 = round((cst / (nb_mutated-1) ),4)

            index_gene_principal = random.choice(indices_mutations)
            genes[index_gene_principal] += cst
            genes[index_gene_principal] = max(0, min(genes[index_gene_principal], PUNTOS_TOTALES))

            for index in indices_mutations:
                if index != index_gene_principal:
                    genes[index] -= cst2
                    genes[index] = max(0, min(genes[index], PUNTOS_TOTALES))
                goal += genes[index]

    new_ch = Character.from_genes([new_type,new_height] + genes[2:]) 
    
    return new_ch


def mutacion_gen(individuo: Character, generation : int, probabilidad_mutacion : float, delta_height : float ,uniform=True):

    prob_mutacion = calculate_probability_mutation(generation,uniform,probabilidad_mutacion)
    genes = individuo.get_genes()
    new_type = genes[0]  # Initialiser avec la valeur actuelle
    new_height = genes[1]
    
    # Mutate the type of character (gene 0)
    # if random.random() < prob_mutacion:
        
    #     random_type = random.choice(list(CharacterType))
    #     new_type = random_type.name
    #     new_ch = Character.from_genes([new_type,new_height] + genes[2:]) 
    #     return new_ch
    
    # Mutate the height (gene 1)
    if random.random() < prob_mutacion:
       
        new_height = genes[1] + random.uniform(-delta_height, delta_height)
        new_height = max(1.3, min(new_height, 2.0))  # Limit height between 1.3 and 2.0 meters
        new_ch = Character.from_genes([new_type,new_height] + genes[2:]) 
        return new_ch
    # no mutation has been done
    return individuo
    
    #here we don t need to modify the other genes cause their sum must be 150 so if we touch one of them 
    #we need the modify an other and we would not doing mutation with 1 gene anymore 
    
