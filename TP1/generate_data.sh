#!/bin/bash

# Definimos el archivo de salida
output_file="resultados.csv"

# Creamos el archivo CSV y añadimos la cabecera
echo "nivel,metodo,heuristica,tiempo,memoria,otros" > "$output_file"

# Iteramos sobre cada nivel
for level in {1..5}
do
  # Iteramos sobre cada método
  for method in bfs dfs astar greedy
  do
    # Si el método es astar o greedy, iteramos sobre las heurísticas
    if [[ "$method" == "astar" ]] || [[ "$method" == "greedy" ]]; then
      for heuristic in manhattan combined
      do
        # Ejecutamos el programa y capturamos la salida
        result=$(python3 main.py --level $level --method $method --heuristic $heuristic)
        
        # Asumimos que el programa imprime el tiempo y la memoria utilizada, que se capturan aquí
        # Esta línea es solo un ejemplo y debe adaptarse según el output de tu programa
        echo "$level,$method,$heuristic,$result" >> "$output_file"
      done
    else
      # Ejecutamos el programa sin heurística y capturamos la salida
      result=$(python3 main.py --level $level --method $method)
      
      # Asumimos que el programa imprime el tiempo y la memoria utilizada
      echo "$level,$method,N/A,$result" >> "$output_file"
    fi
  done
done

echo "Análisis completado. Resultados guardados en $output_file."
