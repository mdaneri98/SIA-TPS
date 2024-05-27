from hopfield import *
from letras import letras
import numpy as np
import matplotlib.pyplot as plt



def salt_pepper(lettre, niveau_bruit):
    bruit = np.random.rand(*lettre.shape) < niveau_bruit
    lettre_bruitee = lettre.copy()
    lettre_bruitee[bruit] = -lettre_bruitee[bruit]
    return lettre_bruitee


def main():
    
    # for letra in letras.values():
    #     imprimir_letra(letra)

    #training
    patterns = [letra.flatten() for letra in letras.values()]
    hopfield_net = HopfieldNetwork(25)
    hopfield_net.train(patterns)

    #mismas lettras
    # mismas_lettras = ['A','F','J','X']
    # for lettra in mismas_lettras:
    #     lettra = letras[lettra].flatten()
    #     imprimir_letra(np.reshape(lettra, (5, 5)))
    #     hopfield_net.recall(lettra,3,True)

    #lettras con ruido
    # ruidas = [0.1,0.2,0.3]
    # for ruida in ruidas : 
    #     lettre_bruitee = salt_pepper(letras['F'].flatten(), ruida)
    #     lettre_bruitee_matrice = np.reshape(lettre_bruitee, (5, 5))
    #     imprimir_letra(lettre_bruitee_matrice)
    #     hopfield_net.recall(lettre_bruitee,5,True)
    
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

    


    





if __name__ == "__main__":
    main()