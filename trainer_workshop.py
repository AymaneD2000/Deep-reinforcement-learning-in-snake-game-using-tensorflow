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
        EXERCICE 3 - ÉQUATION DE BELLMAN POUR Q-LEARNING (★★★ AVANCÉ)
        
        Votre mission : Implémenter l'équation de Bellman qui est au cœur de l'apprentissage par renforcement.
        
        THÉORIE - ÉQUATION DE BELLMAN :
        Q(s,a) = reward + gamma * max(Q(s',a'))
        
        Où :
        - Q(s,a) = Valeur de l'action 'a' dans l'état 's'
        - reward = Récompense immédiate reçue
        - gamma = Facteur de discount (importance des récompenses futures)
        - Q(s',a') = Valeur maximum des actions possibles dans le nouvel état s'
        
        POURQUOI CETTE ÉQUATION ?
        - Elle évalue une action en combinant la récompense immédiate ET le potentiel futur
        - Le gamma contrôle l'importance du futur vs le présent
        - max(Q(s',a')) = la meilleure action possible dans le nouvel état
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
            
            """
            PARTIE À COMPLÉTER - BOUCLE PRINCIPALE :
            
            Pour chaque échantillon dans le batch :
            
            1. SI l'épisode est terminé (done[idx] == True) :
               Q_new = reward[idx]  # Pas de récompense future si le jeu est fini
               
            2. SINON (l'épisode continue) :
               APPLIQUER L'ÉQUATION DE BELLMAN :
               Q_new = reward[idx] + self.gamma * np.max(new_pred[idx])
               
            3. Mettre à jour la target :
               target[idx][np.argmax(action[idx])] = Q_new
            
            EXPLICATION :
            - np.argmax(action[idx]) trouve quelle action a été prise (0, 1, ou 2)
            - On met à jour seulement la Q-value de cette action spécifique
            - Les autres actions gardent leurs anciennes prédictions
            """
            
            for idx in range(len(done)):
                # TODO: Implémentez l'équation de Bellman ici
                
                # Étape 1 : Initialiser Q_new avec la récompense
                Q_new = reward[idx]
                
                # Étape 2 : Si l'épisode n'est pas terminé, ajouter la récompense future
                if not done[idx]: 
                    # TODO: Complétez cette ligne avec l'équation de Bellman
                    # Q_new = reward[idx] + self.gamma * ???
                    pass
                
                # Étape 3 : Mettre à jour la target pour l'action qui a été prise
                # TODO: target[idx][???] = Q_new
                pass
            
            # Calcul de la perte entre les prédictions et les targets
            loss = self.loss_function(target, pred)
        
        # Application de la descente de gradient
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables)) 