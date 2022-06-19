from .base_network import Network

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

class MLPLW(Network):
    def __init__(self, **kwargs):
        super().__init__()
        self.batch_size = kwargs['batch_size']
        self.num_epochs = kwargs['num_epochs']
        self.loss = kwargs['loss']
        self.lr = kwargs['lr']
        self.n_features = kwargs['n_features']
        self.model = self.create_model()

    def create_model(self):
        model = Sequential()
        model.add(Dense(2, activation='relu', kernel_initializer='he_normal', input_shape=(self.n_features,)))
        model.add(Dense(1))
        # compile the model
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.lr
        )
        model.compile(optimizer=optimizer, loss=self.loss)
        return model

    def train(self, x_train, y_train, x_val, y_val):
        if x_val is not None and y_val is not None:
            self.history = self.model.fit(
                x_train,
                y_train,
                batch_size=self.batch_size,
                epochs=self.num_epochs,
                validation_data=(x_val, y_val)
            )
        else:
            self.history = self.model.fit(
                x_train,
                y_train,
                batch_size=self.batch_size,
                epochs=self.num_epochs,
                validation_data=None
            )

    def predict(self,x_test):
        return self.model.predict(x_test)