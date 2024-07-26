import numpy as np
import matplotlib.pyplot as plt
from linear_perceptron import perceptron_training

'''
Implementar el algoritmo de perceptrón simple con función de activación escalón y utilizar el
mismo para aprender los siguientes problemas:

Función lógica “Y” con entradas:
x = {{− 1, 1}, {1, − 1}, {− 1, − 1}, {1, 1}}

y salida esperada:
y = {− 1, − 1, − 1, 1}

Función lógica “O exclusivo” con entradas:
x = {{− 1, 1}, {1, − 1}, {− 1, − 1}, {1, 1}}

y salida esperada:
y = {1, 1, − 1, − 1}

¿Qué puede decir acerca de los problemas que puede resolver el perceptrón simple escalón
en relación a los problemas planteados en la consigna?
'''


def draw_graph_2D(data, weights):
    points = np.array([x for x, _ in data])
    labels = np.array([y for _, y in data])

    # Crear una gráfica de dispersión de los puntos
    plt.scatter(points[:, 0], points[:, 1], c=labels, cmap='bwr', s=100)

    x_values = np.linspace(-2, 2, 100)
    y_values = -(weights[1] / weights[2]) * x_values - (weights[0] / weights[2])  # w1*x + w2*y + b = 0

    # Añadir la línea de decisión a la gráfica
    plt.plot(x_values, y_values, 'g--')

    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.grid(True)
    plt.show()


def graph_dataset_and(learning_rate, limit):
    data_set_and = [
        (np.array([-1, 1]), -1),
        (np.array([1, -1]), -1),
        (np.array([-1, -1]), -1),
        (np.array([1, 1]), 1)
    ]
    _, weights = perceptron_training(data_set_and, learning_rate, limit)
    draw_graph_2D(data_set_and, weights)


def graph_dataset_exclusive_or(learning_rate, limit):
    data_set_or = [
        (np.array([-1, 1]), 1),
        (np.array([1, -1]), 1),
        (np.array([-1, -1]), -1),
        (np.array([1, 1]), -1),
    ]

    _, weights = perceptron_training(data_set_or, learning_rate, limit)
    draw_graph_2D(data_set_or, weights)


def main():
    learning_rate = 0.1
    limit = 100

    graph_dataset_and(learning_rate, limit)
    graph_dataset_exclusive_or(learning_rate, limit)


if __name__ == "__main__":
    main()

'''En el caso de las funciones lógicas AND y XOR, el perceptrón simple de activación
 escalón puede resolver el problema de la función AND, ya que los datos son linealmente separables. 
 Sin embargo, el perceptrón simple no puede resolver el problema de la función XOR, ya que los datos 
 no son linealmente separables. Esto se debe a que la función XOR no puede ser modelada por una única 
 línea recta en el espacio de entrada bidimensional.

En resumen, el perceptrón simple de activación escalón puede resolver problemas
 en los que los datos son linealmente separables, pero no puede resolver problemas donde los
   datos no son linealmente separables, como en el caso de la función XOR.'''