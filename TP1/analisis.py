from main import main
import subprocess
import pandas as pd


methods = ['dfs', 'bfs']
heuristics = ['manhattan', 'combined']
levels = [1, 2, 3]

results = []

for level in levels:
    for method in methods:
        for heuristic in heuristics:
            for _ in range(10):
                result = subprocess.run(['python', 'main.py', '--level', str(level), '--method', method], capture_output=True, text=True)
                output = result.stdout
                output_lines = output.split('\n')
                result_dict = {
                    'Level': level,
                    'Method': method,
                    'Heuristic': heuristic,
                    'Result': output_lines[0],
                    'Cost': output_lines[2].split(': ')[1],
                    'ExploredNodes': output_lines[3].split(': ')[1],
                    'FrontierNodes': output_lines[4].split(': ')[1],
                    'Time': output_lines[5].split(': ')[1],
                }
                results.append(result_dict)

df = pd.DataFrame(results)

df.to_csv('output.csv', index=False)
print("Se guardo el analisis en output.csv")
