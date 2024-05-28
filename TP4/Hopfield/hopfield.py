import numpy as np
from letras import letras


# Función para imprimir una letra
def imprimir_letra(letra):
    for fila in letra:
        print(' '.join(['*' if x == 1 else ' ' for x in fila]))
    


# Clase para la red de Hopfield
class HopfieldNetwork:
    def __init__(self, size):
        self.size = size
        self.weights = np.zeros((size, size))

    def train(self, patterns):
        for p in patterns:
            self.weights += np.outer(p, p)
        self.weights /= len(patterns)
        np.fill_diagonal(self.weights, 0)  
    
    def energy(self,pattern):
        energy = 0
        
        for i in range(24):
            for j in range(24):
                if i < j:
                    energy -= self.weights[i, j] * pattern[i] * pattern[j]
        return energy


    def recall(self, pattern, steps=5, verbose=False):
        energies = []
        pattern = pattern.copy()
        neurons_per_step = self.size // steps   # Seria Si(t)
        energy = self.energy(pattern)
        print(energy)
        energies.append(energy)
        

        for step in range(steps):
            indices = np.random.choice(self.size, neurons_per_step, replace=False)
            for i in indices:
                pattern[i] = np.sign(np.dot(self.weights[i], pattern))

            if verbose:
                print(f"Paso {step + 1}:")
                imprimir_letra(pattern.reshape(5, 5))
            energy = self.energy(pattern)
            print(energy)
            energies.append(energy)
            

        return pattern,energies

    





# Función para agregar ruido a un patrón
def agregar_ruido(pattern, noise_level=1):
    noisy_pattern = pattern.copy()
    indices = np.random.choice(pattern.size, noise_level, replace=False)
    for idx in indices:
        noisy_pattern[idx] = -noisy_pattern[idx]
    return noisy_pattern

