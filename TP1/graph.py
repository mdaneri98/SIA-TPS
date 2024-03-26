import pandas as pd
from main import start
import numpy as np
import utils
import matplotlib.pyplot as plt
import sys

# Suponiendo que la función start y las demás importaciones necesarias ya están definidas

# Define las listas de métodos, heurísticas y niveles
methods = ['dfs', 'bfs', 'greedy', 'astar']
heuristics = ['manhattan', 'combined']
levels = [0, 1, 2, 3]

# Lista para almacenar los resultados
results = []

# Itera a través de cada combinación de nivel, método y heurística
for _ in range(10):
    for level in levels:
        for method in methods:
            # 'greedy' y 'astar' son los únicos métodos que utilizan heurísticas
            if method in ['greedy', 'astar']:
                for heuristic in heuristics:
                    # Llama a start con la combinación actual y almacena el resultado
                    result = start(level, method, heuristic)
                    del result['path']
                    results.append(result)
            else:
                # Métodos que no utilizan heurística
                result = start(level, method, None)  # O 'manhattan' si es necesario pasar una heurística de todos modos
                del result['path']
                result['heuristic'] = 'None'
                results.append(result)

# Convierte la lista de resultados en un DataFrame de pandas
df = pd.DataFrame(results)

# Muestra el DataFrame
print(df)

# Opcional: Guarda el DataFrame en un archivo CSV
df.to_csv('resultados_sokoban.csv', index=False)



# Métodos más rápidos
def plot_average_delta_by_method_and_heuristic_filtered(df):
    # Calculamos el delta promedio para cada combinación de método y heurística
    average_delta = df.groupby(['method', 'heuristic'])['delta'].mean().reset_index()

    # Si algunos métodos no usan heurística, reemplazamos None por un valor más descriptivo para la visualización
    average_delta['heuristic'].fillna('None', inplace=True)

    # Filtramos solo las combinaciones de interés
    filtered = average_delta[
        ((average_delta['method'] == 'dfs') & (average_delta['heuristic'] == 'None')) |
        ((average_delta['method'] == 'bfs') & (average_delta['heuristic'] == 'None')) |
        ((average_delta['method'] == 'astar') & (average_delta['heuristic'] == 'manhattan')) |
        ((average_delta['method'] == 'astar') & (average_delta['heuristic'] == 'combined')) |
        ((average_delta['method'] == 'greedy') & (average_delta['heuristic'] == 'manhattan')) |
        ((average_delta['method'] == 'greedy') & (average_delta['heuristic'] == 'combined'))
    ]

    # Generamos etiquetas para el eje X combinando el método y la heurística
    labels = filtered.apply(lambda row: f"{row['method']} + {row['heuristic']}", axis=1)

    plt.figure(figsize=(10, 6))
    plt.bar(labels, filtered['delta'])
    plt.ylabel('Tiempo Promedio (delta)')
    plt.title('Tiempo Promedio por Método y Heurística (Filtrado)')
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()  # Ajusta automáticamente los parámetros de la subtrama para dar espacio a los ticks
    plt.show()



#Costo promedio según método
def plot_average_cost_by_method(df):
    average_cost = df.groupby('method')['cost'].mean()
    average_cost.plot(kind='bar')
    plt.ylabel('Costo Promedio')
    plt.title('Costo Promedio según Método')
    plt.xticks(rotation=45)
    plt.show()

#Promedio de nodos explorados según método
def plot_average_explored_nodes_by_method(df):
    average_explored_nodes = df.groupby('method')['exploredNodes'].mean()
    average_explored_nodes.plot(kind='bar')
    plt.ylabel('Nodos Explorados Promedio')
    plt.title('Promedio de Nodos Explorados según Método')
    plt.xticks(rotation=45)
    plt.show()



def tiempo_de_resolucion(final_df, nivel):
    level_data = final_df.loc[final_df['level'] == nivel].groupby(['method'], as_index=False)
    mean_data = level_data.mean(numeric_only=True)

    methods = mean_data['method'].tolist()
    delta = mean_data['delta'].tolist()
    std_error_delta = (level_data.std(numeric_only=True)['delta'] / np.sqrt(len(level_data['delta'].mean()['delta']))).tolist()

    plt.figure(figsize=(10,6))
    bars = plt.bar(methods, delta, yerr=std_error_delta, capsize=4, color='#ffca99', width=0.4)

    for bar in bars:
        yval = bar.get_height()  # Obtener el valor de la barra
        plt.text(bar.get_x() + bar.get_width() * 1.25, yval, '{:.6f}'.format(yval), ha='center', va='bottom', fontsize=10)

    plt.title('Tiempo de Resolución para Nivel 3 con cada Método de Búsqueda', fontsize=15)
    plt.xlabel('Método de Búsqueda', fontsize=12)
    plt.ylabel('Tiempo de resolución (s)', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.ylim(0)
    plt.show()

    # -------------------------------------------------------------------------------------

    cost = mean_data['cost'].tolist()
    std_error_cost = (level_data.std(numeric_only=True)['cost'] / np.sqrt(len(level_data['cost'].mean()['cost']))).tolist()


    #Grafico de pasos de resolucion
    plt.figure(figsize=(10,6))
    bars = plt.bar(methods, cost, yerr=std_error_cost, capsize=4, color='#7aeb9a', width=0.4)

    for bar in bars:
        yval = bar.get_height()  # Obtener el valor de la barra
        plt.text(bar.get_x() + bar.get_width()/2.0, yval, str(int(yval)), ha='center', va='bottom', fontsize=10)

    plt.title(f"Cantidad de Pasos de Resolución para Nivel {nivel} con cada Método de Búsqueda", fontsize=15)
    plt.xlabel('Método de Búsqueda', fontsize=12)
    plt.ylabel('Cantidad de pasos', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()


    #Grafico de nodos visitados
    visited_nodes = mean_data['exploredNodes'].tolist()
    std_error_visited = (level_data.std(numeric_only=True)['exploredNodes'] / np.sqrt(len(level_data['exploredNodes'].mean()['exploredNodes']))).tolist()


    plt.figure(figsize=(10,6))
    bars = plt.bar(methods, visited_nodes, yerr=std_error_visited, capsize=4, color='#7ab3eb', width=0.4)

    for bar in bars:
        yval = bar.get_height()  # Obtener el valor de la barra
        plt.text(bar.get_x() + bar.get_width()/2.0, yval, str(int(yval)), ha='center', va='bottom', fontsize=10)

    plt.title(f"Cantidad de Nodos Visitados para Nivel {nivel} con cada Método de Búsqueda", fontsize=15)
    plt.xlabel('Método de Búsqueda', fontsize=12)
    plt.ylabel('Cantidad de Nodos', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()


plot_average_delta_by_method_and_heuristic_filtered(df)
plot_average_cost_by_method(df)
plot_average_explored_nodes_by_method(df)

tiempo_de_resolucion(df, 1)
tiempo_de_resolucion(df, 2)
tiempo_de_resolucion(df, 3)