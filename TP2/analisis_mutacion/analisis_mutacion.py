import matplotlib.pyplot as plt
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Función para leer los datos del archivo CSV y calcular el promedio de rendimiento por generación


#gráfico con la variación del rendimiento promedio para la generación 70 
# (no está mal al parecer) dependiendo de la Proba mutación
# metodo de corte : max gen
# cruze uniform
# Créer un DataFrame à partir du texte




datos_metodos = ['datos_Guerrero1.csv','datos_Guerrero2.csv','datos_Guerrero3.csv','datos_Guerrero4.csv']
names_metodos = ['mutacion_multigen_uniform','mutacion_multigen_no_uniforme','mutacion_gen_uniform','mutacion_gen_no_uniforme']
for dt, name in zip(datos_metodos, names_metodos):
    performances = []

    # Lire le fichier CSV et extraire les performances de la première colonne
    with open(dt, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            performance = float(row[2])  # Assurez-vous d'adapter l'index en fonction de votre fichier CSV
            performances.append(performance)

    # Diviser les performances en groupes de 20 lignes
    groupes_performances = [performances[i:i+20] for i in range(0, len(performances), 20)]

    # Initialiser les listes pour stocker les performances impaires et paires pour chaque groupe
    performances_impaires_groupes = []
    performances_paires_groupes = []

    # Séparer les performances impaires et paires pour chaque groupe
    for groupe in groupes_performances:
        performances_impaires = groupe[::2]
        performances_paires = groupe[1::2]
        performances_impaires_groupes.append(performances_impaires)
        performances_paires_groupes.append(performances_paires)

    # Créer une liste d'abscisses de 0.1 à 1.0 avec un pas de 0.1
    abscisses = np.arange(0.1, 1.1, 0.1)

    # Calculer les positions des boîtes à moustaches pour chaque groupe
    positions_impaires = np.arange(0.1, 1.1, 0.1) + 0.01
    positions_paires = np.arange(0.1, 1.1, 0.1) -0.01



    boxplot1 = plt.boxplot(performances_impaires_groupes, positions=positions_impaires, widths=0.06, patch_artist=True, boxprops=dict(facecolor='blue'), medianprops=dict(color='black'))
    boxplot2 = plt.boxplot(performances_paires_groupes, positions=[pos + 0.05 for pos in positions_paires], widths=0.06, patch_artist=True, boxprops=dict(facecolor='orange'), medianprops=dict(color='black'))
    plt.xlabel('Probabilidad de mutación')
    plt.ylabel('Performance')
    plt.title(f'Gráfico comparativo entre el mejor individuo encontrado después de 70 generaciones \ncon {name} y el mejor individuo de la población inicial')
    plt.xticks([pos + 0.025 for pos in abscisses], ["{:.1f}".format(x) for x in abscisses])
    plt.legend([boxplot1["boxes"][0], boxplot2["boxes"][0]], ['Mejor personaje de la población inicial', 'Mejor personaje encontrado'], loc='lower right')
    plt.grid(True)
    plt.show()









# Fonction pour lire les performances à partir du fichier CSV
def lire_donnees_csv(chemin_fichier):
    performances = []
    with open(chemin_fichier, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            performance = float(row[2])  # Supposons que la performance est dans la troisième colonne (index 2)
            performances.append(performance)
    return performances

def diviser_en_groupes(performances, taille_groupe):
    return [performances[i:i+taille_groupe] for i in range(0, len(performances), taille_groupe)]

def calculer_moyennes(groupes):
    return [np.mean(groupe) for groupe in groupes]

def afficher_graphe_altura(groupes, delta_items):
    plt.boxplot(groupes, positions=delta_items, widths=0.01)
    plt.xlabel('Delta altura')
    plt.ylabel('Performance')
    plt.title('Rendimiento de los mejores individuos encontrados al final de la 70a generación \n en función del delta_altura elegido ')
    plt.grid(True)
    plt.show()

# Liste des valeurs de delta items
delta_items = [0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]

# Chemin vers le fichier CSV
chemin_fichier = 'datos_Guerrero_altura.csv'

# Lire les performances à partir du fichier CSV
performances = lire_donnees_csv(chemin_fichier)

# Diviser les performances en groupes de 50
groupes = diviser_en_groupes(performances, 10)

# Calculer les moyennes pour chaque groupe
moyennes = calculer_moyennes(groupes)

# Afficher le graphe
afficher_graphe_altura(groupes, delta_items)



def afficher_graphe_items(groupes, delta_items):
    plt.boxplot(groupes, positions=delta_items, widths=0.01)
    plt.xlabel('Delta items')
    plt.ylabel('Performance')
    plt.title('Rendimiento de los mejores individuos encontrados al final de la 70a generación \n en función del delta_items elegido ')
    plt.grid(True)
    plt.show()

# Liste des valeurs de delta items
delta_items = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,25,50,75,100]

# Chemin vers le fichier CSV
chemin_fichier = 'datos_Guerrero_items.csv'

# Lire les performances à partir du fichier CSV
performances = lire_donnees_csv(chemin_fichier)

# Diviser les performances en groupes de 50
groupes = diviser_en_groupes(performances, 50)

# Calculer les moyennes pour chaque groupe
moyennes = calculer_moyennes(groupes)

# Afficher le graphe
afficher_graphe_items(groupes, delta_items)