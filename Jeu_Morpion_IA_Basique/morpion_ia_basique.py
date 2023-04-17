# Programme du jeu Morpion

import random

import numpy as np

# Fonction qui demande des paramètres à l'utilisateur
def demande_parametres():
    parametres_ok = False
    while not parametres_ok:
        try:
            joueurs = int(input("Nombre de joueurs (2 ou plus) : "))
            pions = int(input("Nombre de pions à aligner pour gagner (4 ou plus) : "))
            taille = int(input("Taille du plateau : "))
            if joueurs < 2 or pions < 4 or taille < pions:
                print("Veuillez entrer des valeurs valides")
            else:
                parametres_ok = True
        except ValueError:
            print("Veuillez entrer des entiers")
    
    type_joueurs = []
    for i in range(joueurs):
        type_joueur_ok = False
        while not type_joueur_ok:
            type_joueur = input("Type de joueur pour le joueur {} (IA ou Humain) : ".format(i+1))
            if type_joueur == "IA" or type_joueur == "Humain":
                type_joueurs.append(type_joueur)
                type_joueur_ok = True
            else:
                print("Veuillez entrer 'IA' ou 'Humain'")
    return joueurs, pions, taille, type_joueurs


# Fonction qui initialise le plateau
def initialise_plateau(taille):
    plateau = np.zeros((taille,taille))
    return plateau

# Fonction qui affiche le plateau
def affiche_plateau(plateau):
    for i in range(plateau.shape[0]):
        for j in range(plateau.shape[1]):
            if plateau[i][j] == 0:
                print("_", end=" ")
            else:
                print(int(plateau[i][j]), end=" ")
        print("")

# Fonction qui permet à un joueur humain de jouer
def jouer_humain(plateau, joueur):
    coup_ok = False
    while not coup_ok:
        try:
            coup = input("Joueur {} : Entrez 'abandon' pour abandonner, sinon entrez la ligne et la colonne séparées par un espace : ".format(joueur))
            if coup.lower() == "abandon":
                return "abandon", plateau
            else:
                x, y = map(int, coup.split())
                if plateau[x][y] == 0:
                    plateau[x][y] = joueur
                    coup_ok = True
                else:
                    print("Coup non autorisé, réessayez")
        except:
            print("Veuillez entrer des chiffres entre 0 et {} inclus, séparés par un espace".format(plateau.shape[0]-1))
    return "", plateau

# Fonction qui permet à l'IA de jouer
def jouer_IA(plateau, joueur, pions, joueurs):
    x, y = choisir_coup_IA(plateau, joueur, pions, joueurs)
    plateau[x][y] = joueur
    return plateau

# Fonction qui permet à l'IA de choisir un coup
def choisir_coup_IA(plateau, joueur, pions, joueurs):
    # Crée une matrice de zéros de la même taille que le plateau pour stocker les valeurs des coups
    valeurs = np.zeros_like(plateau)

    # Parcourt chaque cellule du plateau
    for i in range(plateau.shape[0]):
        for j in range(plateau.shape[1]):
            # Si la cellule est vide
            if plateau[i][j] == 0:
                # Parcourt toutes les directions possibles (horizontal, vertical, diagonales)
                for direction in [(1, 0), (0, 1), (1, 1), (1, -1)]:
                    # Parcourt toutes les positions possibles pour la séquence de pions
                    for k in range(-pions+1, 1):
                        # Récupère les valeurs des cellules de la séquence
                        sequence = [plateau[i + direction[0] * (k + l), j + direction[1] * (k + l)] for l in range(pions) if
                                      0 <= i + direction[0] * (k + l) < plateau.shape[0] and 0 <= j + direction[1] * (
                                      k + l) < plateau.shape[1]]

                        # Si la séquence a la bonne longueur
                        if len(sequence) == pions:
                            # Si toutes les cellules sont occupées, on ne peut pas jouer ici
                            if 0 not in sequence:
                                valeurs[i][j] = 0
                                break

                            # Parcourt tous les adversaires
                            for adversaire in range(1, joueurs+1):
                                if adversaire != joueur:
                                    # Si l'IA a des pions dans la séquence et pas l'adversaire, incrémente la valeur du coup
                                    if joueur in sequence and adversaire not in sequence:
                                        valeurs[i][j] += (3 ** sequence.count(joueur)) * (sequence.count(0) + 1)
                                    # Si l'adversaire a des pions dans la séquence et pas l'IA, incrémente la valeur du coup
                                    elif adversaire in sequence and joueur not in sequence:
                                        valeurs[i][j] += (5 ** sequence.count(adversaire)) * (sequence.count(0) + 1)

    # Trouve le coup avec la valeur maximale
    max_valeur = np.max(valeurs)
    indices_max = np.argwhere(valeurs == max_valeur)
    index_choisi = random.choice(indices_max)
    
    return index_choisi[0], index_choisi[1]


# Fonction qui détermine si un joueur a gagné
def victoire(plateau, joueur, pions):
        # parcourir les lignes
        for i in range(plateau.shape[0]):
            for j in range(plateau.shape[1]):
                if plateau[i][j] == joueur:
                    nb_pions = 1 # on commence à 1 car on a déjà un pion
                    # on vérifie à droite
                    k = 1
                    while j+k < plateau.shape[1] and plateau[i][j+k] == joueur:
                        nb_pions += 1
                        k += 1
                    # on vérifie à gauche
                    k = 1
                    while j-k >= 0 and plateau[i][j-k] == joueur:
                        nb_pions += 1
                        k += 1
                    # si le joueur a aligné pions pions, il a gagné
                    if nb_pions >= pions:
                        return True
        # parcourir les colonnes
        for i in range(plateau.shape[0]):
            for j in range(plateau.shape[1]):
                if plateau[i][j] == joueur:
                    nb_pions = 1 # on commence à 1 car on a déjà un pion
                    # on vérifie en bas
                    k = 1
                    while i+k < plateau.shape[0] and plateau[i+k][j] == joueur:
                        nb_pions += 1
                        k += 1
                    # on vérifie en haut
                    k = 1
                    while i-k >= 0 and plateau[i-k][j] == joueur:
                        nb_pions += 1
                        k += 1
                    # si le joueur a aligné pions pions, il a gagné
                    if nb_pions >= pions:
                        return True
        # parcourir les diagonales
        for i in range(plateau.shape[0]):
            for j in range(plateau.shape[1]):
                if plateau[i][j] == joueur:
                    nb_pions = 1 # on commence à 1 car on a déjà un pion
                    # on vérifie en bas à droite
                    k = 1
                    while i+k < plateau.shape[0] and j+k < plateau.shape[1] and plateau[i+k][j+k] == joueur:
                        nb_pions += 1
                        k += 1
                    # on vérifie en haut à gauche
                    k = 1
                    while i-k >= 0 and j-k >= 0 and plateau[i-k][j-k] == joueur:
                        nb_pions += 1
                        k += 1
                    # si le joueur a aligné pions pions, il a gagné
                    if nb_pions >= pions:
                        return True
        # parcourir les diagonales inverses
        for i in range(plateau.shape[0]):
            for j in range(plateau.shape[1]):
                if plateau[i][j] == joueur:
                    nb_pions = 1 # on commence à 1 car on a déjà un pion
                    # on vérifie en bas à gauche
                    k = 1
                    while i+k < plateau.shape[0] and j-k >= 0 and plateau[i+k][j-k] == joueur:
                        nb_pions += 1
                        k += 1
                    # on vérifie en haut à droite
                    k = 1
                    while i-k >= 0 and j+k < plateau.shape[1] and plateau[i-k][j+k] == joueur:
                        nb_pions += 1
                        k += 1
                    # si le joueur a aligné pions pions, il a gagné
                    if nb_pions >= pions:
                        return True
        return False

# Fonction qui détermine si le plateau est plein
def match_nul(plateau):
    for i in range(plateau.shape[0]):
        for j in range(plateau.shape[1]):
            if plateau[i][j] == 0:
                return False
    return True

# Programme principal
if __name__ == "__main__":
    joueurs, pions, taille, type_joueurs = demande_parametres()
    plateau = initialise_plateau(taille)
    gagnant = 0
    tour = 1
    abandon = False
    while gagnant == 0 and not match_nul(plateau) and not abandon:
        print("\n")
        print("Tour n°{}".format(tour))
        for i in range(joueurs):
            print("\n")
            affiche_plateau(plateau)
            if type_joueurs[i] == "Humain":
                abandon, plateau = jouer_humain(plateau, i+1)
                if abandon:
                    print("Le joueur {} a abandonné !".format(i+1))
                    break
            elif type_joueurs[i] == "IA":
                plateau = jouer_IA(plateau, i+1, pions, joueurs)
                
            if victoire(plateau, i+1, pions):
                gagnant = i+1
                break
        
        if gagnant != 0 or abandon:
            break
        
        tour += 1
    
    print("\n")
    print("------------Partie terminée !!------------\n")
    affiche_plateau(plateau)
    print("------------------------------------------\n\n")
    if gagnant == 0:
        if abandon:
            print("La partie est terminée à cause d'un abandon.")
        else:
            print("Match nul !")
    else:
        print("Le joueur {} a gagné !".format(gagnant))