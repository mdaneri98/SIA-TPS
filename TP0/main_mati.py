import os
import json
from src.catching import attempt_catch
import matplotlib.pyplot as plt
from src.pokemon import PokemonFactory, StatusEffect
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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

def ejercicio1a_con_barras_sin_desvio():
    pokemons, pokeballs, _ = initialize("config/ejercicio1a-config.json")

    # Creamos y llenamos el DataFrame
    df = pd.DataFrame(columns=['pokemon', 'pokeball', 'success'])
    for pokemon in pokemons:
        for pokeball in pokeballs:
            for _ in range(N):
                success, _ = attempt_catch(pokemon, pokeball)
                new_row = {'pokemon': pokemon.name, 'pokeball': pokeball, 'success': success}
                df.loc[len(df)] = new_row

    # Calcular la media y desvío estándar de atrapadas para cada Pokémon y pokebola.
    media_std_por_pokeball_pokemon = df.groupby(['pokeball', 'pokemon'])['success'].agg(['mean', 'std']).reset_index()
    
    # Crear un gráfico de puntos con barras de error para representar la media y desvío estándar
    fig = px.scatter(media_std_por_pokeball_pokemon, x='pokeball', y='mean', color='pokemon', error_y='std',
                     title='Promedio de Atrapadas por Pokémon y Tipo de Pokeball',
                     labels={'mean': 'Promedio de Atrapadas', 'pokeball': 'Tipo de Pokeball'},
                     hover_data={'mean': True, 'std': True})

    # Mostrar el gráfico
    fig.show()


def ejercicio1a_puntos_con_desvio():
    pokemons, pokeballs, _ = initialize("config/ejercicio1a-config.json")

    # Creamos y llenamos el DataFrame
    df = pd.DataFrame(columns=['pokemon', 'pokeball', 'success'])
    for pokemon in pokemons:
        for pokeball in pokeballs:
            for _ in range(N):
                success, _ = attempt_catch(pokemon, pokeball)
                new_row = {'pokemon': pokemon.name, 'pokeball': pokeball, 'success': success}
                df.loc[len(df)] = new_row

    # Calcular la media y desvío estándar de atrapadas para cada Pokémon y pokebola.
    media_std_por_pokeball_pokemon = df.groupby('pokeball')['success'].agg(['mean', 'std']).reset_index()
    
    # Crear un gráfico de puntos con barras de error para representar la media y desvío estándar
    fig = px.scatter(media_std_por_pokeball_pokemon, x='pokeball', y='mean', error_y='std',
                     title='Promedio de Atrapadas por Pokémon y Tipo de Pokeball',
                     labels={'mean': 'Promedio de Atrapadas', 'pokeball': 'Tipo de Pokeball'},
                     hover_data={'mean': True, 'std': True})

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

    # Agrupar los datos por Pokémon y estado, y calcular la tasa de éxito promedio
    grouped_df = df.groupby(['pokemon', 'status'])['success'].mean().reset_index()
    
    # Crear el gráfico de barras agrupadas
    fig = px.bar(grouped_df, x='status', y='success', color='pokemon',
                title='Efectividad de Captura por Pokémon y Estado',
                barmode='group',
                labels={'success': 'Media de Atrapadas', 'status': 'Estado'})

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



def ejercicio2c():
    factory = PokemonFactory("pokemon.json")
    pokemon_names = ['snorlax', 'caterpie', 'mewtwo']
    pokeballs = ["pokeball", "ultraball", "fastball", "heavyball"]
    status_effects = [StatusEffect.NONE, StatusEffect.POISON, StatusEffect.BURN, StatusEffect.PARALYSIS, StatusEffect.SLEEP, StatusEffect.FREEZE]

    levels = [x for x in range(0, 101)]
    hp_steps = [x * 0.1 for x in range(0, 11)]

    rows = []
    for level in levels:    
        for pokemon_name in pokemon_names:
            for pokeball in pokeballs:
                for status in status_effects:
                    for hp in hp_steps:
                        pokemon = factory.create(pokemon_name, level, status, hp)
                        for _ in range(N):
                            success, _ = attempt_catch(pokemon, pokeball)
                            rows.append({'pokemon': pokemon.name, 'pokeball': pokeball, 'level': level, 'status': status.name, 'hp': hp, 'success': success})
    
    df = pd.DataFrame(rows)

    # Calcular el promedio de atrapadas por nivel
    avg_catch_by_level = df.groupby('level')['success'].mean().reset_index()

    # Calcular el promedio de atrapadas por hp
    avg_catch_by_hp = df.groupby('hp')['success'].mean().reset_index()

    # Calcular el promedio de atrapadas por pokeball
    avg_catch_by_pokeball = df.groupby('pokeball')['success'].mean().reset_index()

    # Calcular el promedio de atrapadas por status
    avg_catch_by_status = df.groupby('status')['success'].mean().reset_index()

    # Crear un solo gráfico de subtramas con los cuatro gráficos de líneas
    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=["Nivel", "HP", "Pokeball", "Status"])

    fig.add_trace(go.Scatter(x=avg_catch_by_level['level'], y=avg_catch_by_level['success'],
                             mode='lines', name='Promedio de Atrapadas por Nivel'), row=1, col=1)

    fig.add_trace(go.Scatter(x=avg_catch_by_hp['hp'], y=avg_catch_by_hp['success'],
                             mode='lines', name='Promedio de Atrapadas por HP'), row=1, col=2)

    fig.add_trace(go.Scatter(x=avg_catch_by_pokeball['pokeball'], y=avg_catch_by_pokeball['success'],
                             mode='lines', name='Promedio de Atrapadas por Pokeball'), row=2, col=1)

    fig.add_trace(go.Scatter(x=avg_catch_by_status['status'], y=avg_catch_by_status['success'],
                             mode='lines', name='Promedio de Atrapadas por Status'), row=2, col=2)

    fig.update_layout(title='Promedio de Atrapadas por Variable')
    fig.update_xaxes(title_text='Nivel', row=1, col=1)
    fig.update_xaxes(title_text='HP', row=1, col=2)
    fig.update_xaxes(title_text='Pokeball', row=2, col=1)
    fig.update_xaxes(title_text='Status', row=2, col=2)
    fig.update_yaxes(title_text='Promedio de Atrapadas', row=1, col=1)
    fig.update_yaxes(title_text='Promedio de Atrapadas', row=2, col=1)

    fig.show()

    # Calcular las diferencias entre la máxima y mínima de los promedios para nivel, hp, pokeball y status
    diff_level = avg_catch_by_level['success'].max() - avg_catch_by_level['success'].min()
    diff_hp = avg_catch_by_hp['success'].max() - avg_catch_by_hp['success'].min()
    diff_pokeball = avg_catch_by_pokeball['success'].max() - avg_catch_by_pokeball['success'].min()
    diff_status = avg_catch_by_status['success'].max() - avg_catch_by_status['success'].min()

    # Graficar el histograma con Plotly
    variables = ['Nivel', 'HP', 'Pokeball', 'Estado']
    diffs = [diff_level, diff_hp, diff_pokeball, diff_status]

    fig = go.Figure(data=[go.Bar(x=variables, y=diffs, marker_color='skyblue')])
    fig.update_layout(title='Diferencia entre Máxima y Mínima de Promedios por Variable',
                      xaxis_title='Variables',
                      yaxis_title='Diferencia entre Máxima y Mínima de Promedios')

    fig.show()

def ejercicio2e(): 
    

    factory = PokemonFactory("pokemon.json")

    with open("config/ejercicio2d-config.json", "r") as file_2d:
        config = json.load(file_2d)
    
    pokemon_name = "onix"
    pokeballs = config["pokeballs"]
    status = config["status"]
    levels = [x for x in range(1, 101, 10)]  # Niveaux de 1 à 100 avec un pas de 10
    hp = config["hp_percentages"]
    
    for level in levels:
        df = pd.DataFrame(columns=['pokemon', 'pokeball', 'hp', 'status', 'success'])
        for pokeball in pokeballs:
            for hp_percentage in hp:
                for status_type in status:
                    pokemon = factory.create(pokemon_name, level, StatusEffect[status_type], hp_percentage)
                    
                    for _ in range(N):
                        success, _ = attempt_catch(pokemon, pokeball)
                        new_row = {'pokemon': pokemon_name, 'pokeball': pokeball, 'hp': hp_percentage, 'status': status_type, 'success': success}
                        df.loc[len(df)] = new_row
    
        mean_success = df.groupby(['pokeball', 'hp', 'status'])['success'].mean().reset_index()
        # Sort by mean success and select top 3 combinations
        top_3_combinations = mean_success.nlargest(3, 'success')
        
        # Create the bar plot
        fig = go.Figure()
        for idx, row in top_3_combinations.iterrows():
            label = f"{row['status']} | HP: {row['hp']} | Ball: {row['pokeball']}"
            fig.add_trace(go.Bar(x=[label], y=[row['success']], name=label))

        # Customize layout
        fig.update_layout(title=f'Top 3 Pokemon Capture Success by Status, HP Percentage, and Pokeball - {level}',
                          xaxis=dict(title='Combination'),
                          yaxis=dict(title='Mean Success'))

        # Show the plot
        fig.show()

if __name__ == "__main__":
    ejercicio2e()