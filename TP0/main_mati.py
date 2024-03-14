import os
import json
from src.catching import attempt_catch
import matplotlib.pyplot as plt
from src.pokemon import PokemonFactory, StatusEffect
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd



N = 100

def initialize(filePath):
    factory = PokemonFactory("pokemon.json")
    
    with open(filePath, "r") as file:
        config = json.load(file)

    LEVEL = config["level"]
    STATUS = StatusEffect[config["status"]]
    HP_PERCENTAGE = config["hp_percentage"]

    jolteon = factory.create("jolteon", LEVEL, STATUS, HP_PERCENTAGE)
    caterpie = factory.create("caterpie", LEVEL, STATUS, HP_PERCENTAGE)
    snorlax = factory.create("snorlax", LEVEL, STATUS, HP_PERCENTAGE)
    onix = factory.create("onix", LEVEL, STATUS, HP_PERCENTAGE)
    mewtwo = factory.create("mewtwo", LEVEL, STATUS, HP_PERCENTAGE)

    pokemons = [jolteon, caterpie, snorlax, onix, mewtwo]
    pokeballs = ["pokeball", "ultraball", "fastball", "heavyball"]
    status_effects = [StatusEffect.NONE, StatusEffect.POISON, StatusEffect.BURN, StatusEffect.PARALYSIS, StatusEffect.SLEEP, StatusEffect.FREEZE]

    return (pokemons, pokeballs, status_effects)

def ejercicio1a():
    pokemons, pokeballs, _ = initialize("config/ejercicio1a-config.json")

    # Creamos y llenamos el DataFrame
    df = pd.DataFrame(columns=['pokemon', 'pokeball', 'success'])
    for pokemon in pokemons:
        for pokeball in pokeballs:
            for _ in range(N):
                success, _ = attempt_catch(pokemon, pokeball)
                new_row = {'pokemon': pokemon.name, 'pokeball': pokeball, 'success': success}
                df.loc[len(df)] = new_row

    # Calcular la media de atrapadas para cada Pokémon y pokebola.
    media_por_pokemon_pokeball = df.groupby(['pokemon', 'pokeball'])['success'].mean().reset_index()
    
    # Media para cada pokemon.
    media_por_pokemon = df.groupby(['pokemon'])['success'].mean().reset_index() 

    # Crear un gráfico de barras con Plotly Express
    fig = px.bar(media_por_pokemon_pokeball, x='pokemon', y='success', color='pokeball', barmode='group', title='Promedio de Atrapadas por Pokémon y Tipo de Pokeball')
    fig.add_trace(go.Scatter(x=media_por_pokemon['pokemon'], y=media_por_pokemon['success']))

    # Mostrar el gráfico
    fig.show()


def ejercicio1a_corregido():
    pokemons, pokeballs, _ = initialize("config/ejercicio1a-config.json")

    # Creamos y llenamos el DataFrame
    df = pd.DataFrame(columns=['pokemon', 'pokeball', 'success'])
    for pokemon in pokemons:
        for pokeball in pokeballs:
            for _ in range(N):
                success, _ = attempt_catch(pokemon, pokeball)
                new_row = {'pokemon': pokemon.name, 'pokeball': pokeball, 'success': success}
                df.loc[len(df)] = new_row

    # Calcular la media de atrapadas para cada Pokémon y pokebola.
    media_por_pokeball_pokemon = df.groupby(['pokeball', 'pokemon'])['success'].mean().reset_index()
    
    # Media para cada pokemon.
    media_por_pokeball = df.groupby(['pokeball'])['success'].mean().reset_index() 

    # Crear un gráfico de barras con Plotly Express
    fig = px.bar(media_por_pokeball_pokemon, x='pokeball', y='success', color='pokemon', barmode='group', title='Promedio de Atrapadas por Pokémon y Tipo de Pokeball')
    fig.add_trace(go.Scatter(x=media_por_pokeball['pokeball'], y=media_por_pokeball['success']))

    # Mostrar el gráfico
    fig.show()


    
def ejercicio1b():
    pokemons, pokeballs, _ = initialize("config/ejercicio1b-config.json")

    # Creamos y llenamos el DataFrame
    df = pd.DataFrame(columns=['pokemon', 'pokeball', 'success'])
    for pokemon in pokemons:
        for pokeball in pokeballs:
            for _ in range(N):
                success, _ = attempt_catch(pokemon, pokeball)
                new_row = {'pokemon': pokemon.name, 'pokeball': pokeball, 'success': success}
                df.loc[len(df)] = new_row

    # Calcular la media de atrapadas para cada Pokémon
    media_por_pokemon = df.groupby('pokemon')['success'].mean().reset_index()

    # Calculamos la media de atrapadas para cada pokemon con la pokebola basica.
    base_pokeball = df[df.pokeball == 'pokeball']
    media_base_pokeball = base_pokeball.groupby('pokemon')['success'].mean()

    media_por_pokemon = pd.DataFrame()
    for pokeball in pokeballs:
        if pokeball != 'pokeball':
            current_ball_serie = df[df.pokeball == pokeball]
            media_current_ball = current_ball_serie.groupby('pokemon')['success'].mean() / media_base_pokeball
            media_current_ball = media_current_ball.reset_index()  # Reiniciar el índice aquí
            if media_por_pokemon.empty:
                media_por_pokemon = media_current_ball
            else:
                media_por_pokemon = media_por_pokemon.merge(media_current_ball, on='pokemon')   # Agregamos la nueva columna ('ultraball' o la que sea)

    # Establecer el índice nuevamente en 'pokemon'
    media_por_pokemon.columns = ['pokemon', 'ultraball', 'fastball', 'heavyball']

    # Graficar
    fig = px.bar(media_por_pokemon, x='pokemon', y=['ultraball', 'fastball', 'heavyball'],
                title='Media de Atrapadas para Cada Pokémon con Diferentes Tipos de Pokeballs',
                labels={'value': 'Media de Atrapadas', 'variable': 'Tipo de Pokeball'}, 
                barmode='group')

    # Mostrar el gráfico
    fig.show()


def ejercicio2a():
    pokemons, pokeballs, status_effects = initialize("config/ejercicio2a-config.json")
    factory = PokemonFactory("pokemon.json")

    df = pd.DataFrame(columns=['pokemon', 'pokeball', 'status', 'success'])
    for pokemon in pokemons:
        for pokeball in pokeballs:
            for current_status in status_effects:
                pokemon = factory.create(pokemon.name, pokemon.level, current_status, 1)
                for _ in range(N):
                    success, _ = attempt_catch(pokemon, pokeball)
                    new_row = {'pokemon': pokemon.name, 'pokeball': pokeball, 'status': current_status.name, 'success': success}
                    df.loc[len(df)] = new_row

#    df = df.groupby(['pokemon', 'status'])['success'].sum()

    # Agrupar los datos por Pokémon y estado, y calcular la tasa de éxito promedio
    grouped_df = df.groupby(['pokemon', 'status'])['success'].mean().reset_index()
    

    # Crear el gráfico de barras agrupadas
    fig = px.bar(grouped_df, x='status', y='success', color='pokemon',
                title='Efectividad de Captura por Pokémon y Estado',
                barmode='group',
                labels={'success': 'Media de Atrapadas', 'status': 'Estado'})

    # Agregar trazos de líneas para cada Pokémon
    for pokemon in grouped_df['pokemon'].unique():
        pokemon_data = grouped_df[grouped_df['pokemon'] == pokemon]
        fig.add_trace(go.Scatter(x=pokemon_data['status'], y=pokemon_data['success'], mode='lines', name=pokemon))

    # Mostrar el gráfico
    fig.show()


def ejercicio2b():
    factory = PokemonFactory("pokemon.json")
    pokeballs = ["pokeball", "ultraball", "fastball", "heavyball"]

    with open("config/ejercicio2b-config.json", "r") as file:
        config = json.load(file)
    
    df_list = []
    hp_steps = [x * 0.1 for x in range(0, 11)]
    for current in config:
        df = pd.DataFrame(columns=['pokemon', 'pokeball', 'hp', 'success'])
        for hp in hp_steps:
            pokemon = factory.create(current['pokemon'], current['level'], StatusEffect[current["status"]], hp)
            
            for _ in range(N):
                for pokeball in pokeballs:
                    success, _ = attempt_catch(pokemon, pokeball)
                    new_row = {'pokemon': pokemon.name, 'pokeball': pokeball, 'hp': hp, 'success': success}
                    df.loc[len(df)] = new_row
            
        # Para un mismo pokemon.
        media_base_por_hp = df[df.pokeball == 'pokeball']['success'].mean()

        # Eficiencia como promedio de la pokebola básica
        media_por_hp = df.groupby('hp')['success'].mean() / media_base_por_hp

        # Agregar los datos al DataFrame de la lista
        df['efficiency'] = df['hp'].map(media_por_hp)
        df_list.append(df)

    # Concatenar todos los DataFrames en uno solo
    final_df = pd.concat(df_list)

    # Graficar con Plotly Express
    fig = px.line(final_df, x='hp', y='efficiency', color='pokemon', line_group='pokemon', labels={'efficiency': 'Eficiencia', 'hp': 'HP', 'pokemon': 'Pokemon'}, title='Eficiencia de captura por HP')
    fig.show()








if __name__ == "__main__":
    ejercicio2b()