# Importation des bibliothèques et modules nécessaires
import numpy as np  # NumPy est utilisé pour les opérations sur les tableaux.
from game import SnakeGameAI, Direction, Point  # Importation des classes nécessaires pour gérer le jeu du serpent.
from collections import deque  # Deque est une structure de données qui permet une manipulation efficace des éléments des deux côtés.
from model import QNetwork  # Importation de la classe QNetwork, définie précédemment, qui représente le modèle de réseau de neurones.
import random  # Random est utilisé pour des choix aléatoires, notamment pour l'exploration par l'agent.
from trainer import QTrainer  # Importation de la classe QTrainer, qui gère l'entraînement du modèle.
import asyncio  # Asyncio est utilisé pour gérer les opérations asynchrones, essentielles pour un entraînement non bloquant.
from helper import plot  # Importation d'une fonction pour tracer les scores de l'agent au fil du temps.

# Définition des constantes utilisées par l'agent
MEMORY_SIZE = 100_000  # Taille maximale de la mémoire où sont stockées les expériences de l'agent.
BATCH_SIZE = 1000  # Taille de l'échantillon utilisé pour l'entraînement sur la mémoire à long terme.
LR = 0.001  # Taux d'apprentissage utilisé par l'optimiseur pour ajuster les poids du modèle.

# Définition de la classe "Agent", qui contrôle le serpent et apprend à jouer au jeu.
class Agent():         
    def __init__(self):
        """
        Initialisation de l'agent.
        """
        # Création du modèle de réseau de neurones, ici avec 256 neurones dans la couche cachée et 3 neurones de sortie (pour les trois actions possibles).
        self.model = QNetwork(256, 3)
        
        # Initialisation des paramètres gamma et epsilon utilisés pour l'apprentissage par renforcement.
        self.gamma = 0.9  # Facteur de discount pour les récompenses futures.
        self.epsilon = 0  # Epsilon détermine la probabilité d'explorer de nouvelles actions au lieu de suivre la politique actuelle.
        
        # Compteur d'itérations du jeu, utilisé pour ajuster epsilon au fil du temps.
        self.game_iteration = 0
        
        # Initialisation de la mémoire de l'agent, qui stocke les expériences récentes sous forme de transitions (état, action, récompense, nouvel état, épisode terminé).
        self.memorys = deque(maxlen=MEMORY_SIZE)
        
        # Création de l'entraîneur qui sera utilisé pour ajuster les poids du modèle en fonction des expériences accumulées.
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
        
    def getState(self, game):
        """
        EXERCICE 1 - REPRÉSENTATION DE L'ÉTAT (★☆☆ DÉBUTANT)
        
        Votre mission : Créer une représentation numérique de l'état du jeu que l'agent peut comprendre.
        
        CONSEILS :
        - L'état doit contenir des informations sur les dangers (collisions potentielles)
        - L'état doit inclure la direction actuelle du serpent  
        - L'état doit indiquer où se trouve la nourriture par rapport à la tête
        
        ÉTAPES À SUIVRE :
        1. Récupérer la position de la tête du serpent : head = game.snake[0]
        
        2. Calculer les positions potentielles si le serpent bouge :
           - point_l = Point(head.x - 20, head.y)  # Gauche
           - point_r = Point(head.x + 20, head.y)  # Droite  
           - point_u = Point(head.x, head.y - 20)  # Haut
           - point_d = Point(head.x, head.y + 20)  # Bas
        
        3. Détecter la direction actuelle du serpent :
           - dir_l = game.direction == Direction.LEFT
           - dir_r = game.direction == Direction.RIGHT
           - dir_u = game.direction == Direction.UP
           - dir_d = game.direction == Direction.DOWN
        
        4. Créer un tableau 'state' avec 11 éléments booléens :
           - [0-2] : Dangers (droit devant, à droite, à gauche)
           - [3-6] : Direction du serpent (gauche, droite, haut, bas) 
           - [7-10] : Position relative de la nourriture (gauche, droite, haut, bas)
        
        AIDE pour les dangers :
        - Danger droit devant : le serpent va-t-il dans un mur/lui-même s'il continue ?
        - Danger à droite : y a-t-il un danger si le serpent tourne à droite ?
        - Danger à gauche : y a-t-il un danger si le serpent tourne à gauche ?
        
        Utilisez game.is_collision(point) pour tester les collisions.
        
        RETOUR ATTENDU : return np.array(state, dtype=int)
        """
        
        # TODO: Implémentez votre code ici
        pass

    def memory(self, state, action, reward, next_state, done):
        """
        Sauvegarde l'expérience actuelle dans la mémoire de l'agent.

        :param state: État actuel du jeu.
        :param action: Action effectuée par l'agent.
        :param reward: Récompense reçue après l'action.
        :param next_state: Nouvel état du jeu après l'action.
        :param done: Booléen indiquant si l'épisode est terminé (par exemple, si le serpent est mort).
        """
        self.memorys.append((state, action, reward, next_state, done))  # Ajout de l'expérience à la mémoire.

    async def train_short_memory(self, state, action, reward, next_state, done):
        """
        Entraînement à court terme, utilisé après chaque étape du jeu.

        :param state: État actuel du jeu.
        :param action: Action effectuée par l'agent.
        :param reward: Récompense reçue après l'action.
        :param next_state: Nouvel état du jeu après l'action.
        :param done: Booléen indiquant si l'épisode est terminé.
        """
        await asyncio.sleep(0)  # Permet au système de gérer d'autres tâches, utile pour la gestion asynchrone.
        await self.trainer.trainer(state, action, reward, next_state, done)  # Entraînement basé sur cette unique expérience.

    async def train_long_memory(self):
        """
        Entraînement à long terme, utilisé après plusieurs étapes du jeu, en échantillonnant aléatoirement la mémoire.

        """
        # Si la mémoire contient plus d'expériences que BATCH_SIZE, on en sélectionne un échantillon aléatoire.
        if BATCH_SIZE < len(self.memorys):
            sample = random.sample(self.memorys, BATCH_SIZE)
        else:
            # Sinon, on utilise toutes les expériences disponibles.
            sample = self.memorys
            
        # Décomposition de l'échantillon en éléments séparés (état, action, récompense, etc.).
        state, action, reward, next_state, done = zip(*sample)
        
        await asyncio.sleep(0)  # Permet au système de gérer d'autres tâches.
        await self.trainer.trainer(state, action, reward, next_state, done)  # Entraînement basé sur cet échantillon.

    def getAction(self, state):
        """
        EXERCICE 2 - STRATÉGIE EPSILON-GREEDY (★★☆ INTERMÉDIAIRE)
        
        Votre mission : Implémenter la stratégie epsilon-greedy qui équilibre exploration et exploitation.
        
        CONCEPTS CLÉS :
        - EXPLORATION : Essayer des actions aléatoires pour découvrir de nouvelles stratégies
        - EXPLOITATION : Utiliser le modèle entraîné pour choisir la meilleure action
        - EPSILON : Probabilité de faire de l'exploration (diminue avec le temps)
        
        ÉTAPES À SUIVRE :
        
        1. Calculer epsilon (exploration décroissante) :
           self.epsilon = 50 - self.game_iteration
           
        2. Initialiser l'action : final_move = [0, 0, 0]
           - [1,0,0] = continuer tout droit
           - [0,1,0] = tourner à droite  
           - [0,0,1] = tourner à gauche
        
        3. Décision exploration vs exploitation :
           
           SI random.randint(0, 100) < self.epsilon :
               # EXPLORATION - Action aléatoire
               move = random.randint(0, 2)
               final_move[move] = 1
               
           SINON :
               # EXPLOITATION - Utiliser le modèle
               a) Formater l'état : state0 = np.expand_dims(np.array(state, dtype=int), axis=0)
               b) Prédiction du modèle : prediction = self.model(state0).numpy()
               c) Choisir la meilleure action : move = np.argmax(prediction)
               d) Activer l'action : final_move[move] = 1
        
        POURQUOI EPSILON-GREEDY ?
        - Au début : beaucoup d'exploration (epsilon élevé) pour apprendre
        - Avec le temps : plus d'exploitation (epsilon diminue) pour optimiser
        
        RETOUR ATTENDU : return final_move
        """
        
        # TODO: Implémentez votre code ici
        pass

# Fonction principale pour entraîner l'agent.
async def train():
    """
    Fonction pour entraîner l'agent à jouer au jeu du serpent. 
    """
    # Initialisation des variables pour suivre et tracer les scores au fil du temps.
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    
    # Création de l'agent et du jeu.
    agent = Agent()
    game = SnakeGameAI()
    
    while True:
        # Récupération de l'état actuel du jeu.
        state_old = agent.getState(game)

        # L'agent choisit une action en fonction de l'état actuel.
        final_move = agent.getAction(state_old)
        
        # Le jeu avance d'une étape selon l'action choisie, et retourne la récompense, un indicateur de fin, et le score.
        reward, done, score = game.play_step(final_move)
        
        # Récupération du nouvel état après l'action.
        state_new = agent.getState(game)
        
        # Entraînement à court terme avec la transition actuelle.
        await agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # Stockage de la transition dans la mémoire de l'agent.
        agent.memory(state_old, final_move, reward, state_new, done)

        if done:  # Si l'épisode est terminé (le serpent est mort).
            # Réinitialisation du jeu pour un nouvel épisode.
            game.reset()
            agent.game_iteration += 1  # Incrémente le compteur d'itérations.
            
            # Entraînement à long terme sur un échantillon de la mémoire.
            await agent.train_long_memory()

            # Mise à jour du record et des statistiques.
            if score > record:
                record = score
                agent.model.save()  # Sauvegarde du modèle si un nouveau record est atteint.

            print('Game', agent.game_iteration, 'Score', score, 'Record:', record)

            # Mise à jour des listes pour le tracé des scores.
            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.game_iteration
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)  # Affichage du graphique des scores.

if __name__ == '__main__':
    import asyncio
    asyncio.run(train()) 