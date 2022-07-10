from .base_network import Network
from .layers import *

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

class ResidualMLP(Network):
    def __init__(self, **kwargs):
        super().__init__()
        self.batch_size = kwargs['batch_size']
        self.num_epochs = kwargs['num_epochs']
        self.loss = kwargs['loss']
        self.lr = kwargs['lr']
        self.n_features = kwargs['n_features']
        self.num_epochs_overfit = kwargs['num_epochs_overfit']
        self.model = self.create_model()

    def create_model(self):
        x_input = tf.keras.Input(shape=(self.n_features,))
        x_dense1 = layers.Dense(6, kernel_initializer='he_normal')(x_input)
        x_act1 = layers.PReLU()(x_dense1)
        x_dense2 = layers.Dense(8, kernel_initializer='he_normal')(tf.concat([x_act1, x_input], -1))
        x_act2 = layers.PReLU()(x_dense2)
        x_dense3 = layers.Dense(6, kernel_initializer='he_normal')(tf.concat([x_act2, x_act1], -1))
        x_act3 = layers.PReLU()(x_dense3)
        x_dense4 = layers.Dense(6, kernel_initializer='he_normal')(tf.concat([x_act3, x_act2], -1))
        x_act4 = layers.PReLU()(x_dense4)
        x_dense5 = layers.Dense(4, kernel_initializer='he_normal')(tf.concat([x_act4, x_act3], -1))
        x_act5 = layers.PReLU()(x_dense5)
        x_dense6 = layers.Dense(6, kernel_initializer='he_normal')(x_act5)
        x_act6 = layers.PReLU()(x_dense6)
        x_dense7 = layers.Dense(4, kernel_initializer='he_normal')(x_act6)
        x_act7 = layers.PReLU()(x_dense7)
        x_dense8 = layers.Dense(2, kernel_initializer='he_normal')(x_act7)
        x_act8 = layers.PReLU()(x_dense8)
        out = layers.Dense(1)(x_act8)
        model = tf.keras.Model(x_input, out)

        # compile the model
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.lr
        )
        model.compile(optimizer=optimizer, loss=self.loss)
        model.summary()
        return model

    def train(self, x_train, y_train, x_val, y_val):
        num_epochs = self.num_epochs if len(x_train) != 1 else self.num_epochs_overfit
        if x_val is not None and y_val is not None:
            self.history = self.model.fit(
                x_train,
                y_train,
                batch_size=self.batch_size,
                epochs=num_epochs,
                validation_data=(x_val, y_val)
            )
        else:
            self.history = self.model.fit(
                x_train,
                y_train,
                batch_size=self.batch_size,
                epochs=num_epochs,
                validation_data=None
            )

    def predict(self,x_test):
        return self.model.predict(x_test)

    def to_json(self):
        return self.model.to_json()

class GRN(Network):
    def __init__(self, **kwargs):
        super().__init__()
        self.batch_size = kwargs['batch_size']
        self.num_epochs = kwargs['num_epochs']
        self.loss = kwargs['loss']
        self.lr = kwargs['lr']
        self.n_features = kwargs['n_features']
        self.num_epochs_overfit = kwargs['num_epochs_overfit']
        self.model = self.create_model()

    def create_model(self):
        x_input = tf.keras.Input(shape=(self.n_features,))
        x = GatedResidualNetwork(5, 0.1)(x_input)
        x = GatedResidualNetwork(5, 0.1)(x)
        x = GatedResidualNetwork(5, 0.1)(x)
        x = GatedResidualNetwork(5, 0.1)(x)
        x = Dense(3)(x)
        x = layers.PReLU()(x)
        x = Dense(2)(x)
        x = layers.PReLU()(x)
        out = Dense(1)(x)
        model = tf.keras.Model(x_input, out)

        # compile the model
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.lr
        )
        model.compile(optimizer=optimizer, loss=self.loss)
        model.summary()
        return model

    def train(self, x_train, y_train, x_val, y_val):
        num_epochs = self.num_epochs if len(x_train) != 1 else self.num_epochs_overfit
        if x_val is not None and y_val is not None:
            self.history = self.model.fit(
                x_train,
                y_train,
                batch_size=self.batch_size,
                epochs=num_epochs,
                validation_data=(x_val, y_val)
            )
        else:
            self.history = self.model.fit(
                x_train,
                y_train,
                batch_size=self.batch_size,
                epochs=num_epochs,
                validation_data=None
            )

    def predict(self,x_test):
        return self.model.predict(x_test)

    def to_json(self):
        return self.model.to_json()