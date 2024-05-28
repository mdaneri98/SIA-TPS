from hopfield import *
from letras import letras
import numpy as np
import matplotlib.pyplot as plt



def salt_pepper(lettre, niveau_bruit):
    bruit = np.random.rand(*lettre.shape) < niveau_bruit
    lettre_bruitee = lettre.copy()
    lettre_bruitee[bruit] = -lettre_bruitee[bruit]
    return lettre_bruitee



def comparer_patterns(pattern1, pattern2, patterns):
    if np.array_equal(pattern1, pattern2):
        return 1
    
    for p in patterns:
        if np.array_equal(pattern1, p):
            return 0
    else : 
        return -1

def fonction_ultima(n_essais):
    ruidos = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]  # Niveaux de bruit testés
    patterns = [letra.flatten() for letra in letras.values()]
    # Initialisation des listes pour stocker les résultats
    tp_ruido, fp_ruido, tn_ruido = [], [], []
    hopfield_net = HopfieldNetwork(25)
    hopfield_net.train(patterns)


    for ruido in ruidos:
        tp, fp, tn = 0, 0, 0
        for _ in range(n_essais):
            lettre_bruitee = salt_pepper(letras['F'].flatten(), ruido)
            lettre_reconnue, _ = hopfield_net.recall(lettre_bruitee.flatten(), 10)
            res = comparer_patterns(lettre_reconnue, letras['F'].flatten(), patterns)
            if res == 1:
                tp += 1
            elif res == 0:
                fp += 1
            else:
                tn += 1

        # Ajout des résultats aux listes
        tp_ruido.append(tp)
        fp_ruido.append(fp)
        tn_ruido.append(tn)

    # Conversion en proportions si nécessaire
    tp_ruido = np.array(tp_ruido)
    fp_ruido = np.array(fp_ruido)
    tn_ruido = np.array(tn_ruido)
    total_cases = n_essais

    # Paramètres pour les barres groupées
    bar_width = 0.25  # Largeur des barres
    r1 = np.arange(len(ruidos))  # Positions pour les tp
    r2 = [x + bar_width for x in r1]  # Positions pour les fp
    r3 = [x + bar_width for x in r2]  # Positions pour les tn

    # Création du graphique en barres groupées
    plt.figure(figsize=(12, 6))
    plt.bar(r1, tp_ruido, color='green', width=bar_width, edgecolor='grey', label='Positive')
    plt.bar(r2, fp_ruido, color='yellow', width=bar_width, edgecolor='grey', label='Fake Positive')
    plt.bar(r3, tn_ruido, color='red', width=bar_width, edgecolor='grey', label='Negative')

    # Ajout des labels et des titres
    plt.xlabel('Nivel de ruido', fontweight='bold')
    plt.ylabel('Cantidad de casos', fontweight='bold')
    plt.title('El efecto del nivel de ruido en los resultados', fontweight='bold')
    plt.xticks([r + bar_width for r in range(len(ruidos))], ruidos)
    plt.legend()

    # Affichage du graphique
    plt.show()



def main():
    
    for letra in letras.values():
        imprimir_letra(letra)

    #training
    patterns = [letra.flatten() for letra in letras.values()]
    hopfield_net = HopfieldNetwork(25)
    hopfield_net.train(patterns)

    #mismas lettras
    mismas_lettras = ['A','F','J','X']
    for lettra in mismas_lettras:
        lettra = letras[lettra].flatten()
        imprimir_letra(np.reshape(lettra, (5, 5)))
        hopfield_net.recall(lettra,3,True)


    #lettras con ruido
    ruidas = [0.1,0.2,0.3]
    for ruida in ruidas : 
        lettre_bruitee = salt_pepper(letras['F'].flatten(), ruida)
        lettre_bruitee_matrice = np.reshape(lettre_bruitee, (5, 5))
        imprimir_letra(lettre_bruitee_matrice)
        hopfield_net.recall(lettre_bruitee,5,True)
    
    energies = [[] for _ in range(10)]

    # Exécution de la fonction recall 10 fois et stockage des énergies
    for i in range(10):
        lettre_bruitee = salt_pepper(letras['A'].flatten(), 0.4)
        _, energies[i] = hopfield_net.recall(lettre_bruitee, 10, False)

    # Calcul de la moyenne des énergies à chaque étape
    moyenne_energies = [np.mean([e[i] for e in energies]) for i in range(10)]

    # Tracé du graphique de l'évolution moyenne de l'énergie
    plt.plot(moyenne_energies)
    plt.xlabel('Paso')
    plt.ylabel('Energy')
    plt.title('Evolución media de la energía en 10 ejecuciones de la letra A')
    plt.show()

    

    fonction_ultima(500)









    





if __name__ == "__main__":
    main()