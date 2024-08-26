# Importation des bibliothèques nécessaires
import tensorflow as tf
import os

class QNetwork(tf.keras.Model):
    def __init__(self, hidden_size, output_size):
        super(QNetwork, self).__init__()
        self.dense1 = tf.keras.layers.Dense(hidden_size, activation='relu')
        self.dense2 = tf.keras.layers.Dense(output_size)

    def call(self, input):
        x = self.dense1(input)
        return self.dense2(x)
    
    def save(self, file_name='best10.weights.h5'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        file_name = os.path.join(model_folder_path, file_name)
        self.save_weights(file_name)

