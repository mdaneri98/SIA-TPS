# TP2 SIA - Grupo 6

## Integrantes
- Daneri, Matias
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

| Sección              | Parámetro             | Descripción                                                                                                         | Valores soportados                                                                                                                                          |
|----------------------|-----------------------|---------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [Población]          | cantidad_poblacion    | Tamaño de la población                                                                                              | Número entero                                                                                                                                               |
| [Población]          | k                     | Cantidad de individuos seleccionados para la reproducción                                                           | Número entero                                                                                                                                               |
| [Población]          | tipo                  | Tipo de personaje                                                                                                   | 'guerrero', 'mago', 'arquero', 'infiltrado'                                                                                                                 |
| [Selección]          | metodo1               | Método de selección                                                                                                 | 'seleccion_ruleta', 'seleccion_ranking', 'seleccion_torneo_deterministico', 'torneo_probabilistico', 'seleccion_universal', 'seleccion_boltzmann'                                                             |
| [Selección]          | metodo2               | Método de selección                                                                                                 | 'seleccion_ruleta', 'seleccion_ranking', 'seleccion_torneo_deterministico', 'torneo_probabilistico', 'seleccion_universal', 'seleccion_boltzmann'                                                             |
| [Selección]          | metodo3               | Método de selección                                                                                                 | 'seleccion_ruleta', 'seleccion_ranking', 'seleccion_torneo_deterministico', 'torneo_probabilistico', 'seleccion_universal', 'seleccion_boltzmann'                                                             |
| [Selección]          | metodo4               | Método de selección                                                                                                 | 'seleccion_ruleta', 'seleccion_ranking', 'seleccion_torneo_deterministico', 'torneo_probabilistico', 'seleccion_universal', 'seleccion_boltzmann'                                                             |
| [Selección]          | a                     | Determina el número de k individuos a generar de los métodos de seleccion 1 y 2                                     | Número entero                                                                                                                                               |
| [Selección]          | b                     | Determina el número de k individuos a generar de los métodos de seleccion 3 y 4                                     | Número entero                                                                                                                                               |
| [Selección]          | m                     | En caso de haber elegido torneo_deterministico, esta sería la cantidad de individuos para cada torneo               | Número entero                                                                                                                                               |
| [Selección]          | threshold             | En caso de haber elegido torneo_probabilistico, este sería el valor del threshold a utilizar                        | Número decimal entre 0.5 y 1                                                                                                                                |
| [Selección]          | temperatura_inicial   | En caso de haber elegido boltzmann, este sería el valor de la temperatura inicial                                   | Número decimal entre 0 y 1                                                                                                                                  |
| [Crossover]          | metodo                | Método de crossover                                                                                                 | 'cruce_un_punto', 'cruce_dos_puntos', 'cruce_anular', 'cruce_uniforme'                                                                                      |
| [Crossover]          | probability           | Probabilidad de crossover                                                                                           | Número decimal entre 0 y 1                                                                                                                                  |
| [Condicion de corte] | time_limit            | Tiempo de fin de la ejecución                                                                                       | Formato HH:MM:SS                                                                                                                                            |
| [Condicion de corte] | metodo                | Método de parada                                                                                                    | 'cantidad_generaciones'                                                                                                                                     |
| [Condicion de corte] | max_generations       | Cantidad máxima de generaciones a ejecutar sin que se produzca un cambio generacional antes de detener el algoritmo | Número entero                                                                                                                                               |
| [Condicion de corte] | optimal_fitness       | Valor de fitness óptimo                                                                                             | Número entero                                                                                                                                               |
| [Condicion de corte] | optimal_fitness_error | Error permitido para el fitness óptimo                                                                              | Número entero                                                                                                                                               |
| [Condicion de corte] | delta                 | Valor máximo de diferencia para considerar que dos valores son distintos                                            | Número decimal entre 0 y 1                                                                                                                                  |
| [Mutación]           | metodo1               | Método de mutación                                                                                                  | 'mutacion_multigen_uniform', 'mutacion_multigen_no_uniforme', 'mutacion_gen_uniform', 'mutacion_gen_no_uniforme'|
| [Mutación]           | metodo2               | Método de mutación                                                                                                  | 'mutacion_multigen_uniform', 'mutacion_multigen_no_uniforme', 'mutacion_gen_uniform', 'mutacion_gen_no_uniforme'|
| [Mutación]           | metodo3               | Método de mutación                                                                                                  | 'mutacion_multigen_uniform', 'mutacion_multigen_no_uniforme', 'mutacion_gen_uniform', 'mutacion_gen_no_uniforme'|
| [Mutación]           | metodo4               | Método de mutación                                                                                                  | 'mutacion_multigen_uniform', 'mutacion_multigen_no_uniforme', 'mutacion_gen_uniform', 'mutacion_gen_no_uniforme'|
| [Mutación]           | probabilidad_mutacion | Probabilidad de mutación                                                                                            | Número decimal entre 0 y 1                                                                                                                                  |
| [Mutación]           | delta_items           | Tipo de gen a mutar                                                                                                 | Número entero                                                                                                                                               |
| [Mutación]           | delta_height          | Cantidad de genes a mutar                                                                                           | Número entero                                                                                                                                               |