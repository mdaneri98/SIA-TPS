# TP2 SIA - Grupo 6

## Integrantes
- Daneria, Matias
- Flores, Magdalena
- Limachi, Desiree
- Rouquette, Joseph

## Introducción

Se implementó un programa en Python con el objetivo de generar un motor de algoritmos genéticos para obtener las mejores configuraciones de personajes de un juego de rol.

### Requisitos

- Python3 (versión 3.8.5 o superior)
- pip3
- pipenv

### Instalación

En la carpeta del tp2 ejecutar.
```sh
pipenv install
```
para instalar las dependencias necesarias en el ambiente virtual.

## Ejecución

Para ejecutar el programa se deberá posicionar en la carpeta raíz del proyecto: 
```python
pipenv shell
python main.py
```

Debido a la gran cantidad de parámetros posibles para el programa, se optó por incluir un archivo de configuración arguments.config en lugar de enviar los parámetros por línea de comandos.

### Configuración

A continuación se detallan los posibles parámetros para la ejecución del programa:

| Parámetro              | Descripción                                                                                                         | Valores soportados                                                                                                                                          |
|------------------------|---------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------|
| n                      | Tamaño de la población                                                                                              | Número entero                                                                                                                                               |
| k                      | Cantidad de individuos seleccionados para la reproducción                                                           | Número entero                                                                                                                                               |
| tipo                   | Tipo de personaje                                                                                                   | 'guerrero', 'mago', 'arquero', 'infiltrado'                                                                                                                 |
| selection_method_name1 | Método de selección                                                                                                 | 'ruleta', 'ranking', 'torneo_deterministico', 'torneo_probabilistico', 'universal', 'boltzmann'                                                             |
| selection_method_name2 | Método de selección                                                                                                 | 'ruleta', 'ranking', 'torneo_deterministico', 'torneo_probabilistico', 'universal', 'boltzmann'                                                             |
| selection_method_name3 | Método de selección                                                                                                 | 'ruleta', 'ranking', 'torneo_deterministico', 'torneo_probabilistico', 'universal', 'boltzmann'                                                             |
| selection_method_name4 | Método de selección                                                                                                 | 'ruleta', 'ranking', 'torneo_deterministico', 'torneo_probabilistico', 'universal', 'boltzmann'                                                             |
| A                      | Determina el número de k individuos a generar de los métodos de seleccion 1 y 2                                     | Número entero                                                                                                                                               |
| B                      | Determina el número de k individuos a generar de los métodos de seleccion 3 y 4                                     | Número entero                                                                                                                                               |
| M                      | En caso de haber elegido torneo_deterministico, esta sería la cantidad de individuos para cada torneo               | Número entero                                                                                                                                               |
| threshold              | En caso de haber elegido torneo_probabilistico, este sería el valor del threshold a utilizar                        | Número decimal entre 0.5 y 1                                                                                                                                |
| temperatura_inicial    | En caso de haber elegido boltzmann, este sería el valor de la temperatura inicial                                   | Número decimal entre 0 y 1                                                                                                                                  |
| crossover_method_name  | Método de crossover                                                                                                 | 'cruce_un_punto', 'cruce_dos_puntos', 'cruce_anular', 'cruce_uniforme'                                                                                      |
| crossover_probability  | Probabilidad de crossover                                                                                           | Número decimal entre 0 y 1                                                                                                                                  |
| start_time             | Tiempo de inicio de la ejecución                                                                                    | Formato HH:MM:SS                                                                                                                                            |
| time_limit             | Tiempo de fin de la ejecución                                                                                       | Formato HH:MM:SS                                                                                                                                            |
| stop_method            | Método de parada                                                                                                    | 'cantidad_generaciones'                                                                                                                                     |
| max_generations        | Cantidad máxima de generaciones a ejecutar sin que se produzca un cambio generacional antes de detener el algoritmo | Número entero                                                                                                                                               |
| optimal_fitness        | Valor de fitness óptimo                                                                                             | Número entero                                                                                                                                               |
| optimal_fitness_error  | Error permitido para el fitness óptimo                                                                              | Número entero                                                                                                                                               |
| delta                  | Valor máximo de diferencia para considerar que dos valores son distintos                                            | Número decimal entre 0 y 1                                                                                                                                  |
| selection_mut_name1    | Método de mutación                                                                                                  | 'seleccion_boltzmann', 'seleccion_ruleta', 'seleccion_ranking', 'seleccion_universal', 'seleccion_torneo_deterministico', 'seleccion_torneo_probabilistico' |
| selection_mut_name2    | Método de mutación                                                                                                  | 'seleccion_boltzmann', 'seleccion_ruleta', 'seleccion_ranking', 'seleccion_universal', 'seleccion_torneo_deterministico', 'seleccion_torneo_probabilistico' |
| selection_mut_name3    | Método de mutación                                                                                                  | 'seleccion_boltzmann', 'seleccion_ruleta', 'seleccion_ranking', 'seleccion_universal', 'seleccion_torneo_deterministico', 'seleccion_torneo_probabilistico' |
| selection_mut_name4    | Método de mutación                                                                                                  | 'seleccion_boltzmann', 'seleccion_ruleta', 'seleccion_ranking', 'seleccion_universal', 'seleccion_torneo_deterministico', 'seleccion_torneo_probabilistico' |
| probabilidad_mutacion  | Probabilidad de mutación                                                                                            | Número decimal entre 0 y 1                                                                                                                                  |
| delta_items            | Tipo de gen a mutar                                                                                                 | Número entero                                                                                                                                               |
| delta_height           | Cantidad de genes a mutar                                                                                           | Número entero                                                                                                                                               |