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
        state = np.array(state, dtype=int)
        next_state = np.array(next_state, dtype=int)
        action = np.array(action, dtype=int)
        reward = np.array(reward, dtype=np.float16)
        
        if len(state.shape) == 1:
            state = np.expand_dims(state, axis=0)
            next_state = np.expand_dims(next_state, axis=0)
            action = np.expand_dims(action, axis=0)
            reward = np.expand_dims(reward, axis=0)
            done = (done,)
        with tf.GradientTape() as tape:
            pred = self.model(state)
            new_pred = self.model(next_state)
            target = np.copy(pred)
            for idx in range(len(done)):
                Q_new = reward[idx]
                if not done[idx]: 
                    Q_new = reward[idx] + self.gamma * np.max(new_pred[idx])
                target[idx][np.argmax(action[idx])] = Q_new
            
            loss = self.loss_function(target, pred)
        
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))