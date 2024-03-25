import pandas as pd
from main import start
import utils
import sys

# Suponiendo que la función start y las demás importaciones necesarias ya están definidas

# Define las listas de métodos, heurísticas y niveles
methods = ['dfs', 'bfs', 'greedy', 'astar']
heuristics = ['manhattan', 'combined']
levels = [0, 1, 2, 3]

# Lista para almacenar los resultados
results = []

# Itera a través de cada combinación de nivel, método y heurística
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
            results.append(result)

# Convierte la lista de resultados en un DataFrame de pandas
df = pd.DataFrame(results)

# Muestra el DataFrame
print(df)

# Opcional: Guarda el DataFrame en un archivo CSV
df.to_csv('resultados_sokoban.csv', index=False)
