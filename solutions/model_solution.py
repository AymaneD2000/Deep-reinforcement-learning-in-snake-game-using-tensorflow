# SOLUTION COMPLÈTE - RÉSEAU DE NEURONES Q-NETWORK

import tensorflow as tf
import os

class QNetwork(tf.keras.Model):
    """
    SOLUTION EXERCICE 4 - ARCHITECTURE DU RÉSEAU DE NEURONES COMPLÈTE
    """
    
    def __init__(self, hidden_size, output_size):
        super(QNetwork, self).__init__()
        
        # SOLUTION : Définition des couches du réseau
        
        # Couche cachée avec activation ReLU
        self.dense1 = tf.keras.layers.Dense(hidden_size, activation='relu')
        
        # Couche de sortie sans activation (pour les Q-values)
        self.dense2 = tf.keras.layers.Dense(output_size)

    def call(self, input):
        """
        SOLUTION : PROPAGATION AVANT (Forward Pass)
        
        Cette fonction définit comment les données passent à travers le réseau :
        État (11 valeurs) → Couche cachée (ReLU) → Q-values (3 valeurs)
        """
        
        # Passer l'input dans la première couche (couche cachée)
        x = self.dense1(input)
        
        # Passer le résultat dans la deuxième couche (couche de sortie)
        return self.dense2(x)
    
    def save(self, file_name='best10.weights.h5'):
        """
        Sauvegarde les poids du modèle entrainé.
        Cette fonction est déjà complète - pas besoin de la modifier.
        """
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        file_name = os.path.join(model_folder_path, file_name)
        self.save_weights(file_name) 