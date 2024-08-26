# Importation des bibliothèques nécessaires
import pygame  # Pygame est utilisé pour gérer l'interface graphique du jeu.
import random  # Random est utilisé pour générer des positions aléatoires pour la nourriture.
from enum import Enum  # Enum permet de créer des énumérations, ici pour les directions.
from collections import namedtuple  # Namedtuple est utilisé pour créer des points avec des coordonnées x et y.
import numpy as np  # NumPy est utilisé pour manipuler les actions du serpent.

# Initialisation de Pygame et de la police d'écriture
pygame.init()
font = pygame.font.Font('arial.ttf', 25)

# Définition de la classe Direction pour les quatre directions possibles du serpent
class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

# Définition d'une structure pour représenter un point (x, y)
Point = namedtuple('Point', 'x, y')

# Définition des couleurs RGB utilisées dans le jeu
WHITE = (255, 255, 255)  # Couleur blanche, utilisée pour le texte
RED = (200, 0, 0)  # Couleur rouge, utilisée pour la nourriture
BLUE1 = (0, 0, 255)  # Bleu foncé, utilisé pour le corps du serpent
BLUE2 = (0, 100, 255)  # Bleu clair, utilisé pour l'intérieur du corps du serpent
BLACK = (0, 0, 0)  # Couleur noire, utilisée pour l'arrière-plan

# Taille des blocs qui composent le serpent et la nourriture
BLOCK_SIZE = 20
# Vitesse de rafraîchissement du jeu (nombre d'images par seconde)
SPEED = 40

# Définition de la classe SnakeGameAI qui gère le jeu du serpent
class SnakeGameAI:
    
    def __init__(self, w=640, h=480):
        """
        Initialise les paramètres du jeu.

        :param w: Largeur de la fenêtre du jeu.
        :param h: Hauteur de la fenêtre du jeu.
        """
        self.w = w
        self.h = h
        self.display = pygame.display.set_mode((self.w, self.h))  # Création de la fenêtre du jeu
        pygame.display.set_caption('Snake')  # Titre de la fenêtre
        self.clock = pygame.time.Clock()  # Horloge pour gérer le taux de rafraîchissement
        self.reset()  # Initialisation du jeu

    def reset(self):
        """
        Réinitialise le jeu à son état initial.
        """
        self.direction = Direction.RIGHT  # Direction initiale du serpent (droite)
        self.head = Point(self.w / 2, self.h / 2)  # Position initiale de la tête du serpent (centre de la fenêtre)
        self.snake = [self.head,
                      Point(self.head.x - BLOCK_SIZE, self.head.y),
                      Point(self.head.x - (2 * BLOCK_SIZE), self.head.y)]  # Corps initial du serpent (trois blocs)

        self.score = 0  # Score initial
        self.food = None  # Position de la nourriture (sera définie par _place_food)
        self._place_food()  # Place la nourriture à une position aléatoire
        self.frame_iteration = 0  # Compteur de frames (utilisé pour limiter le nombre d'actions possibles)

    def _place_food(self):
        """
        Place la nourriture à un endroit aléatoire sur la grille.
        """
        x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE  # Coordonnée x aléatoire
        y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE  # Coordonnée y aléatoire
        self.food = Point(x, y)  # Position de la nourriture
        if self.food in self.snake:  # Si la nourriture est placée sur le serpent, on la replace
            self._place_food()

    def play_step(self, action):
        """
        Avance le jeu d'une étape en fonction de l'action choisie par l'agent.

        :param action: Action choisie par l'agent (avancer, tourner à droite, tourner à gauche).
        :return: Tuple contenant la récompense, l'état de fin de partie (True/False), et le score actuel.
        """
        self.frame_iteration += 1  # Incrémente le compteur de frames
        # 1. Collecte des événements (comme la fermeture de la fenêtre)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # 2. Déplacement du serpent
        self._move(action)  # Mise à jour de la position de la tête du serpent
        self.snake.insert(0, self.head)  # Ajout de la nouvelle position de la tête au début de la liste représentant le corps du serpent

        # 3. Vérification des collisions (si le serpent se mange ou touche les murs)
        reward = -0.1  # Récompense par défaut (léger coût pour chaque action)
        game_over = False
        if self.is_collision() or self.frame_iteration > 100 * len(self.snake):
            game_over = True  # Fin du jeu en cas de collision ou de dépassement du nombre maximum de frames
            reward = -10  # Récompense négative importante pour une collision
            return reward, game_over, self.score

        # 4. Vérification si le serpent a mangé la nourriture
        if self.head == self.food:
            self.score += 1  # Incrémente le score
            reward = 10  # Récompense positive pour avoir mangé la nourriture
            self._place_food()  # Place une nouvelle nourriture
        else:
            self.snake.pop()  # Supprime le dernier élément du corps du serpent (le serpent avance)

        # 5. Mise à jour de l'interface graphique et de l'horloge
        self._update_ui()
        self.clock.tick(SPEED)  # Contrôle de la vitesse du jeu

        # 6. Retourne la récompense, l'état de fin de partie et le score
        return reward, game_over, self.score

    def is_collision(self, pt=None):
        """
        Vérifie si le serpent entre en collision avec les murs ou avec lui-même.

        :param pt: Point à vérifier (par défaut la tête du serpent).
        :return: True si le serpent est en collision, sinon False.
        """
        if pt is None:
            pt = self.head  # Si aucun point n'est spécifié, on vérifie la tête du serpent
        # Vérifie si la tête touche les limites de la fenêtre
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        # Vérifie si la tête touche une partie du corps du serpent
        if pt in self.snake[1:]:
            return True

        return False

    def _update_ui(self):
        """
        Met à jour l'interface graphique avec les nouvelles positions du serpent et de la nourriture.
        """
        self.display.fill(BLACK)  # Remplit l'écran de noir

        # Dessine chaque segment du serpent
        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12))  # Dessine un petit carré bleu clair au centre de chaque segment

        # Dessine la nourriture
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        # Affiche le score en haut à gauche de l'écran
        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()  # Rafraîchit l'écran

    def _move(self, action):
        """
        Déplace le serpent dans la direction choisie par l'agent.

        :param action: Tableau indiquant l'action choisie [avancer, tourner à droite, tourner à gauche].
        """
        # Liste des directions dans le sens des aiguilles d'une montre
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)  # Indice de la direction actuelle

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]  # Pas de changement de direction
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4  # Tourne à droite
            new_dir = clock_wise[next_idx]
        else:  # [0, 0, 1]
            next_idx = (idx - 1) % 4  # Tourne à gauche
            new_dir = clock_wise[next_idx]

        self.direction = new_dir  # Met à jour la direction du serpent

        # Met à jour la position de la tête du serpent en fonction de la direction choisie
        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)  # Met à jour la position de la tête
