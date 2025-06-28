# SOLUTIONS COMPLÈTES - AGENT D'APPRENTISSAGE PAR RENFORCEMENT

import numpy as np
from game import SnakeGameAI, Direction, Point
from collections import deque
from model import QNetwork
import random
from trainer import QTrainer
import asyncio
from helper import plot

MEMORY_SIZE = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent():         
    def __init__(self):
        self.model = QNetwork(256, 3)
        self.gamma = 0.9
        self.epsilon = 0
        self.game_iteration = 0
        self.memorys = deque(maxlen=MEMORY_SIZE)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
        
    def getState(self, game):
        """
        SOLUTION EXERCICE 1 - REPRÉSENTATION DE L'ÉTAT
        """
        # Position de la tête du serpent
        head = game.snake[0]
        
        # Calcul des positions potentielles si le serpent se déplace
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
        # Vérification de la direction actuelle du serpent
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        # Création du tableau d'état avec 11 éléments booléens
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
            
            # Position relative de la nourriture par rapport à la tête
            game.food.x < game.head.x,  # Nourriture à gauche
            game.food.x > game.head.x,  # Nourriture à droite
            game.food.y < game.head.y,  # Nourriture en haut
            game.food.y > game.head.y   # Nourriture en bas
        ]

        return np.array(state, dtype=int)

    def memory(self, state, action, reward, next_state, done):
        self.memorys.append((state, action, reward, next_state, done))

    async def train_short_memory(self, state, action, reward, next_state, done):
        await asyncio.sleep(0)
        await self.trainer.trainer(state, action, reward, next_state, done)

    async def train_long_memory(self):
        if BATCH_SIZE < len(self.memorys):
            sample = random.sample(self.memorys, BATCH_SIZE)
        else:
            sample = self.memorys
            
        state, action, reward, next_state, done = zip(*sample)
        await asyncio.sleep(0)
        await self.trainer.trainer(state, action, reward, next_state, done)

    def getAction(self, state):
        """
        SOLUTION EXERCICE 2 - STRATÉGIE EPSILON-GREEDY
        """
        # Réduction de l'exploration au fil du temps
        self.epsilon = 50 - self.game_iteration
        
        # Initialisation de l'action finale
        final_move = [0, 0, 0]
        
        # Exploration : action aléatoire avec probabilité epsilon
        if random.randint(0, 100) < self.epsilon:
            move = random.randint(0, 2)  # Action aléatoire
            final_move[move] = 1
        else:
            # Exploitation : utiliser le modèle pour prédire la meilleure action
            state0 = np.expand_dims(np.array(state, dtype=int), axis=0)
            prediction = self.model(state0).numpy()
            move = np.argmax(prediction)  # Action avec la plus haute Q-value
            final_move[move] = 1
            
        return final_move

async def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    
    agent = Agent()
    game = SnakeGameAI()
    
    while True:
        state_old = agent.getState(game)
        final_move = agent.getAction(state_old)
        reward, done, score = game.play_step(final_move)
        state_new = agent.getState(game)
        
        await agent.train_short_memory(state_old, final_move, reward, state_new, done)
        agent.memory(state_old, final_move, reward, state_new, done)

        if done:
            game.reset()
            agent.game_iteration += 1
            await agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print('Game', agent.game_iteration, 'Score', score, 'Record:', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.game_iteration
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)

if __name__ == '__main__':
    import asyncio
    asyncio.run(train()) 