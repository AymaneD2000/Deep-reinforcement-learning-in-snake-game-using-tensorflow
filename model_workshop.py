# Importation des bibliothèques nécessaires
import tensorflow as tf
import os

class QNetwork(tf.keras.Model):
    """
    EXERCICE 4 - ARCHITECTURE DU RÉSEAU DE NEURONES (★★☆ INTERMÉDIAIRE)
    
    Votre mission : Créer un réseau de neurones pour approximer la fonction Q(état, action).
    
    CONTEXTE - DEEP Q-LEARNING :
    - Le réseau prend en entrée l'état du jeu (11 valeurs)
    - Il produit en sortie les Q-values pour chaque action possible (3 actions)
    - L'agent choisit l'action avec la plus haute Q-value
    
    ARCHITECTURE RECOMMANDÉE :
    - Couche d'entrée : 11 neurones (taille de l'état)
    - Couche cachée : hidden_size neurones avec activation ReLU
    - Couche de sortie : output_size neurones (3 actions) sans activation
    
    POURQUOI ReLU ?
    - ReLU(x) = max(0, x) : simple et efficace
    - Évite le problème du gradient qui disparaît
    - Accélère l'entraînement
    
    POURQUOI PAS D'ACTIVATION EN SORTIE ?
    - Les Q-values peuvent être négatives ou positives
    - On veut la valeur brute de la prédiction
    """
    
    def __init__(self, hidden_size, output_size):
        super(QNetwork, self).__init__()
        
        """
        TODO: Définissez les couches du réseau
        
        Utilisez tf.keras.layers.Dense pour créer :
        
        1. self.dense1 = Couche cachée avec :
           - hidden_size neurones
           - activation='relu'
           
        2. self.dense2 = Couche de sortie avec :
           - output_size neurones  
           - pas d'activation (par défaut)
        
        AIDE :
        tf.keras.layers.Dense(nombre_neurones, activation='fonction')
        """
        
        # TODO: Implémentez vos couches ici
        pass

    def call(self, input):
        """
        PROPAGATION AVANT (Forward Pass)
        
        Cette fonction définit comment les données passent à travers le réseau.
        
        TODO: Implémentez la propagation avant :
        
        1. x = Passer l'input dans self.dense1
        2. return Passer x dans self.dense2
        
        Le réseau transforme : État (11 valeurs) → Couche cachée → Q-values (3 valeurs)
        """
        
        # TODO: Implémentez la propagation avant
        pass
    
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