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
        Fonction pour obtenir l'état actuel du jeu à partir des informations sur le serpent et la nourriture.

        :param game: Instance du jeu SnakeGameAI en cours.
        :return: Un tableau représentant l'état du jeu sous forme d'indicateurs (par exemple, danger devant, direction du serpent, emplacement de la nourriture).
        """
        # Position de la tête du serpent.
        head = game.snake[0]
        
        # Calcul des positions potentielles si le serpent se déplace vers la gauche, la droite, le haut ou le bas.
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
        # Vérification de la direction actuelle du serpent.
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        # Création du tableau d'état, qui indique les dangers, la direction du serpent, et la position de la nourriture.
        state = [
            # Danger droit devant
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            # Danger à droite
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            # Danger à gauche
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),
            
            # Direction actuelle du serpent
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Position relative de la nourriture par rapport à la tête du serpent
            game.food.x < game.head.x,  # Nourriture à gauche
            game.food.x > game.head.x,  # Nourriture à droite
            game.food.y < game.head.y,  # Nourriture en haut
            game.food.y > game.head.y  # Nourriture en bas
        ]

        return np.array(state, dtype=int)  # Conversion de la liste en tableau NumPy pour compatibilité avec le modèle.

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
        Détermine l'action que l'agent doit prendre à partir de l'état actuel du jeu.

        :param state: État actuel du jeu.
        :return: Action à effectuer, représentée par un tableau binaire (par exemple, [1, 0, 0] pour aller à gauche).
        """
        # Réduction de l'exploration au fil du temps (epsilon décroît avec les itérations du jeu).
        self.epsilon = 50 - self.game_iteration
        
        # Initialisation de l'action finale (par exemple, [0, 0, 0] signifie pas de mouvement).
        final_move = [0, 0, 0]
        
        # Exploration : l'agent choisit une action aléatoire avec une probabilité dépendant de epsilon.
        if random.randint(0, 100) < self.epsilon:
            move = random.randint(0, 2)  # Choisit une action aléatoire parmi les trois possibles.
            final_move[move] = 1  # Active cette action.
        else:
            # Exploitation : l'agent choisit l'action avec la meilleure prédiction du modèle.
            state0 = np.expand_dims(np.array(state, dtype=int), axis=0)  # Reformate l'état pour compatibilité avec le modèle.
            prediction = self.model(state0).numpy()  # Obtenu la prédiction du modèle.
            move = np.argmax(prediction)  # Choisit l'action avec la valeur prédite la plus élevée.
            final_move[move] = 1  # Active cette action.
            
        return final_move  # Retourne l'action choisie.

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
            agent.game_iteration += 1  # Incrémentation du compteur d'itérations du jeu.
            
            # Entraînement à long terme après plusieurs transitions.
            await agent.train_long_memory()

            if score > record:  # Mise à jour du record si un nouveau score record est atteint.
                record = score
                agent.model.save()  # Sauvegarde du modèle si un nouveau record est atteint.

            print('Game', agent.game_iteration, 'Score', score, 'Record:', record)  # Affichage des statistiques actuelles.

            # Mise à jour des scores pour les graphiques.
            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.game_iteration
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)  # Traçage des scores.

# Démarrage de l'entraînement si le script est exécuté directement.
if __name__ == '__main__':
    asyncio.run(train())  # Lancement de la boucle d'entraînement asynchrone.
