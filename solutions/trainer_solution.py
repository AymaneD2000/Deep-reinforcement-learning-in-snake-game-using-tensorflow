# SOLUTION COMPLÈTE - TRAINER AVEC ÉQUATION DE BELLMAN

import tensorflow as tf
import numpy as np

class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.loss_function = tf.keras.losses.MeanSquaredError()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)

    async def trainer(self, state, action, reward, next_state, done):
        """
        SOLUTION EXERCICE 3 - ÉQUATION DE BELLMAN COMPLÈTE
        """
        
        # Préparation des données d'entrée
        state = np.array(state, dtype=int)
        next_state = np.array(next_state, dtype=int)
        action = np.array(action, dtype=int)
        reward = np.array(reward, dtype=np.float16)
        
        # Gestion des dimensions pour traiter un ou plusieurs échantillons
        if len(state.shape) == 1:
            state = np.expand_dims(state, axis=0)
            next_state = np.expand_dims(next_state, axis=0)
            action = np.expand_dims(action, axis=0)
            reward = np.expand_dims(reward, axis=0)
            done = (done,)
            
        # Calcul des gradients avec TensorFlow
        with tf.GradientTape() as tape:
            # Prédictions actuelles du modèle pour les états actuels
            pred = self.model(state)
            
            # Prédictions du modèle pour les nouveaux états (après action)
            new_pred = self.model(next_state)
            
            # Copie des prédictions actuelles (seront modifiées avec les nouvelles Q-values)
            target = np.copy(pred)
            
            # BOUCLE PRINCIPALE - APPLICATION DE L'ÉQUATION DE BELLMAN
            for idx in range(len(done)):
                # Initialiser Q_new avec la récompense
                Q_new = reward[idx]
                
                # Si l'épisode n'est pas terminé, ajouter la récompense future
                if not done[idx]: 
                    # ÉQUATION DE BELLMAN : Q_new = reward + gamma * max(Q_next_state)
                    Q_new = reward[idx] + self.gamma * np.max(new_pred[idx])
                
                # Mettre à jour la target pour l'action qui a été prise
                target[idx][np.argmax(action[idx])] = Q_new
            
            # Calcul de la perte entre les prédictions et les targets
            loss = self.loss_function(target, pred)
        
        # Application de la descente de gradient
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables)) 