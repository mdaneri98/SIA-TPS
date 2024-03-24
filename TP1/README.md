# TP1 SIA - Grupo 6

#Integrantes
- Daneria, Matias
- Flores, Magdalena
- Limachi, Desiree
- Rouquette, Joseph

## Introducción

Se implementó un programa en Python para encontrar soluciones con diferentes métodos de búsqueda del juego [Sokoban](http://www.game-sokoban.com/index.php?mode=level&lid=200).

### Requisitos

- Python3
- 

## Ejecución

Para ejecutar el mismo se deberá posicionar en la carpeta raíz del proyecto: 
```
python main.py --level 2 --method bfs 
```

Significado de cada uno de los parámetros: 
| Parámetro |  Descripción | Valores soportados |
|----       | ------------------ | ------------------ |
| --level             | Indica el nivel a resolver. | [1-5]|
| --method            | Indica el método de búsqueda que utilizará para resolver el tablero.  | bfs, dfs, astar, greedy|
| --heuristic         | Indica la heurística que se utilizará.  | manhattan, combined|

