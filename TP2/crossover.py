from character import Character
from math import ceil
import random


class Crossover:



    @staticmethod
    def cruce_un_punto(first_parent: Character, second_parent: Character) -> tuple:
        # Obtener los genes de cada padre
        genes1 = first_parent.get_genes()
        genes2 = second_parent.get_genes()
        
        # Elegir un punto de cruce al azar dentro del rango más pequeño de items de ambos padres
        min_len = min(len(genes1), len(genes2))
        punto_cruce = random.randint(0, min_len - 1)

        # Efectuamos la cruza de genes.
        child_genes1 = genes1[:punto_cruce] + genes2[punto_cruce:]
        child_genes2 = genes2[:punto_cruce] + genes1[punto_cruce:]
        
        child1 = Character.from_genes(child_genes1)
        child2 = Character.from_genes(child_genes2)

        return child1, child2


    @staticmethod
    def cruce_dos_puntos(first_parent: Character, second_parent: Character) -> tuple:
        # Obtener los genes de cada padre
        genes1 = first_parent.get_genes()
        genes2 = second_parent.get_genes()
        
        # Elegimos dos locus al azar dentro del rango más pequeño de genes de ambos padres.
        min_len = min(len(genes1), len(genes2))

        punto_cruce1 = random.randint(0, min_len - 1)
        punto_cruce2 = random.randint(0, min_len - 1)

        cruces = sorted([punto_cruce1, punto_cruce2])

        # Efectuamos la cruza de genes.
        child_genes1 = genes1[:cruces[0]] + genes2[cruces[0]:cruces[1]] + genes1[cruces[1]:]
        child_genes2 = genes2[:cruces[0]] + genes1[cruces[0]:cruces[1]] + genes2[cruces[1]:]
        
        child1 = Character.from_genes(child_genes1)
        child2 = Character.from_genes(child_genes2)

        return child1, child2
    

    @staticmethod
    def cruce_anular(first_parent: Character, second_parent: Character) -> tuple:
        '''
            Este método realiza un cruce anular entre dos padres para producir dos hijos.
            Se selecciona un punto de cruce al azar y una longitud para el segmento de genes a intercambiar.
            Si la suma del punto de cruce y la longitud excede el tamaño del gen, el intercambio continúa desde el inicio del gen.

            ## Ejemplo.
            
            len(gen1) = 12
            P = 11 && L = 5 

            0 1 1 0 1 1 0 0 1 1 1 0
                                  ^
                    ^

            1 0 1 0 0 1 1 1 0 0 1 0
        
            Hay que tener en cuenta si P + L excede la longitud de la lista de genes -> intercambiar los del inicio.
        '''
        start_index = 2

        # Obtener los genes de cada padre
        genes1 = first_parent.get_genes()
        genes2 = second_parent.get_genes()

        # Elegir un punto de cruce al azar dentro del rango más pequeño de items de ambos padres
        len_genes = min(len(genes1), len(genes2))
        long = random.randint(0, ceil(len_genes/2))
        punto_cruce = random.randint(0, len_genes - 1)

        # Calcular el punto de finalización del cruce
        end_index = start_index + long

        # Si end_index es mayor que la longitud de los genes, envolver alrededor
        if end_index > len_genes:
            end_index %= len_genes
            # Realizar el cruce anular
            genes1[start_index:], genes2[start_index:] = genes2[start_index:], genes1[start_index:]
            genes1[:end_index], genes2[:end_index] = genes2[:end_index], genes1[:end_index]
        else:
            # Realizar el cruce sin envolver alrededor
            genes1[start_index:end_index], genes2[start_index:end_index] = genes2[start_index:end_index], genes1[start_index:end_index]

        child1 = Character.from_genes(genes1)
        child2 = Character.from_genes(genes2)

        return child1, child2
    

    @staticmethod
    def cruce_uniforme(first_parent: Character, second_parent: Character, probability: float) -> tuple:
        if probability < 0 or probability > 1:
            raise Exception("Probability must be a float in range(0,1)")
        
        # Obtener los genes de cada padre
        genes1 = first_parent.get_genes()
        genes2 = second_parent.get_genes()

        child_genes1 = genes1.copy()
        child_genes2 = genes2.copy()

        min_len = min(len(genes1), len(genes2))
        change = [random.choice([True, False]) for _ in range(min_len)]

        for i in range(min_len):
            if change[i]:
                aux = genes1[i]
                genes1[i] = genes2[i]
                genes2[i] = aux

        child1 = Character.from_genes(child_genes1)
        child2 = Character.from_genes(child_genes2)

        return child1, child2