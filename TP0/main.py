from src.catching import attempt_catch
from src.pokemon import PokemonFactory, StatusEffect
import numpy as np
import matplotlib.pyplot as plt
from src.catching import attempt_catch

def ejercicio2a():
    factory = PokemonFactory("pokemon.json")
    pokemon_names = ["snorlax", "jolteon", "caterpie", "onix", "mewtwo"]
    pokeball_names = ["heavyball", "ultraball", "fastball", "pokeball"]
    status_effects = [StatusEffect.NONE, StatusEffect.POISON, StatusEffect.BURN, StatusEffect.PARALYSIS, StatusEffect.SLEEP, StatusEffect.FREEZE]

    N = 100

    fig, ax = plt.subplots()

    bar_width = 0.2
    index = np.arange(len(pokemon_names))

    for i, pokeball_name in enumerate(pokeball_names):
        probabilities_hp_avg = []
        for pokemon_name in pokemon_names:
            probabilities_hp = []
            for status_effect in status_effects:
                for _ in range(N):
                    success, probability = attempt_catch(factory.create(pokemon_name, 100, status_effect, 1), pokeball_name)
                    probabilities_hp.append(probability)
            probabilities_hp_avg.append(np.mean(probabilities_hp))

        ax.bar(index + i * bar_width, probabilities_hp_avg, bar_width, label=pokeball_name)

    ax.set_xlabel('Pokemon')
    ax.set_ylabel('Probabilidad de Captura Promedio')
    ax.set_title('Probabilidad de Captura Promedio por Pokemon y Pokebola')
    ax.set_xticks(index + 0.6)
    ax.set_xticklabels(pokemon_names)
    ax.legend()

    plt.tight_layout()
    plt.show()


    
def ejercicio2b():
    factory = PokemonFactory("pokemon.json")
    pokeball_names = ["heavyball", "ultraball", "fastball", "pokeball"]

    N = 100
    hp_values = np.arange(0.0, 1.01, 0.2)  # Valores de puntos de vida (0.0 a 1.0, dividido en 0.01)

    fig, ax = plt.subplots()
    pokemon_name = "snorlax"

    status_effect = StatusEffect.NONE  # Usar un solo estado de salud

    bar_width = 0.2
    index = np.arange(len(hp_values))

    for i, pokeball_name in enumerate(pokeball_names):
        probabilities_hp_avg = []
        for hp_percentage in hp_values:
            probabilities_hp = []
            for _ in range(N):
                success, probability = attempt_catch(factory.create(pokemon_name, 100, status_effect, hp_percentage), pokeball_name)
                probabilities_hp.append(probability)
            probabilities_hp_avg.append(np.mean(probabilities_hp))

        ax.bar(index + i * bar_width, probabilities_hp_avg, bar_width, label=pokeball_name)

    ax.set_xlabel('Porcentaje de Puntos de Vida')
    ax.set_ylabel('Probabilidad de Captura Promedio')
    ax.set_title(f'Efecto de los Puntos de Vida en la Probabilidad de Captura - {pokemon_name.capitalize()}')
    ax.set_xticks(index + 0.3)
    ax.set_xticklabels([f"{int(hp*100)}%" for hp in hp_values])
    ax.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    ejercicio2b()