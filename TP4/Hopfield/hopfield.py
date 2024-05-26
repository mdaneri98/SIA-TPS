import numpy as np
from letras import letras


# Función para imprimir una letra
def imprimir_letra(letra):
    for fila in letra:
        print(' '.join(['*' if x == 1 else ' ' for x in fila]))
    print("\n")


# Clase para la red de Hopfield
class HopfieldNetwork:
    def __init__(self, size):
        self.size = size
        self.weights = np.zeros((size, size))

    def train(self, patterns):
        for p in patterns:
            self.weights += np.outer(p, p)
        self.weights /= len(patterns)
        np.fill_diagonal(self.weights, 0)  # No tener conexiones propias

    def recall(self, pattern, steps=5, verbose=False):
        pattern = pattern.copy()
        neurons_per_step = self.size // steps   # Seria Si(t)

        for step in range(steps):
            indices = np.random.choice(self.size, neurons_per_step, replace=False)
            for i in indices:
                pattern[i] = np.sign(np.dot(self.weights[i], pattern))

            if verbose:
                print(f"Paso {step + 1}:")
                imprimir_letra(pattern.reshape(5, 5))

        return pattern


# Función para agregar ruido a un patrón
def agregar_ruido(pattern, noise_level=1):
    noisy_pattern = pattern.copy()
    indices = np.random.choice(pattern.size, noise_level, replace=False)
    for idx in indices:
        noisy_pattern[idx] = -noisy_pattern[idx]
    return noisy_pattern


def main():
    # Convertir las letras a vectores
    patterns = [letra.flatten() for letra in letras.values()]

    # Crear y entrenar la red de Hopfield
    hopfield_net = HopfieldNetwork(25)
    hopfield_net.train(patterns)

    # Solicitar al usuario que elija una letra
    letra = input("Ingrese la letra que desea utilizar (A, B, C, D): ").upper()
    if letra not in letras:
        print("Letra no válida. Por favor, elija una letra válida.")
        return

    # Crear un patrón ruidoso de la letra elegida
    original_pattern = letras[letra].flatten()
    noisy_pattern = agregar_ruido(original_pattern, noise_level=5)

    # Imprimir patrón ruidoso
    print("Patrón ruidoso inicial:")
    imprimir_letra(noisy_pattern.reshape(5, 5))

    # Recuperar el patrón ruidoso
    output_pattern = hopfield_net.recall(noisy_pattern, steps=5, verbose=True)

    # Verificar si el patrón recuperado es un estado espurio
    es_estado_espurio = True
    for pattern in patterns:
        if np.array_equal(output_pattern, pattern):
            print("El patrón recuperado coincide con uno de los patrones originales.")
            es_estado_espurio = False
            break

    if es_estado_espurio:
        print("El patrón recuperado es un estado espurio.")

    # Imprimir el patrón recuperado final
    print("Patrón recuperado final:")
    imprimir_letra(output_pattern.reshape(5, 5))


if __name__ == "__main__":
    main()
