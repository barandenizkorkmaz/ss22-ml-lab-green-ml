import numpy as np
import tensorflow as tf
from base_network import Network


# TODO: Add val dataset
class SimpleRNN(Network):
    def __init__(self, units = 32, hidden_dense=16, ragged=False, mask=True):
        super(SimpleRNN, self).__init__()
        self.ragged = ragged
        self.model = tf.keras.models.Sequential()
        if mask:
            self.model.add(tf.keras.layers.Masking())
        self.model.add(tf.keras.layers.LSTM(units, return_sequences=False))
        self.model.add(tf.keras.layers.Dense(hidden_dense, activation='relu'))
        self.model.add(tf.keras.layers.Dense(1, activation='relu'))
        self.history = None

    def train(self, x, y, loss, optimizer, epochs):
        if not self.ragged:
            x = tf.keras.preprocessing.sequence.pad_sequences(x, padding='post')
            seq_len, feat_size = x[0].shape
        else:
            seq_len = None
            feat_size = x[0].shape[1]
        self.model.build(input_shape=(None, seq_len, feat_size)) # Should I replace None w/ num_models?
        self.model.compile(loss=loss, optimizer=optimizer)
        self.history = self.model.fit(x=x, y=y, epochs=epochs)
        return self.history

    def predict(self, x):
        return self.model.predict(x)


class EmbeddedRNN(Network):
    def __init__(self, num_ops, seq_len, feat_size, out_dim=10, units=32, hidden_dense=16):
        super(EmbeddedRNN, self).__init__()
        self.input = tf.keras.Input(shape=(seq_len, feat_size))
        self.x = tf.keras.layers.Embedding(num_ops, out_dim)(self.input[:, :, 0])
        print(self.x.shape)
        print(self.input[:,:,1:].shape)
        self.x = tf.keras.layers.Concatenate()((self.x, self.input[:, :, 1:]))
        self.x = tf.keras.layers.LSTM(units, return_sequences=False, input_shape=(seq_len, out_dim+6))(self.x)
        self.x = tf.keras.layers.Dense(hidden_dense, activation='relu')(self.x)
        self.output = tf.keras.layers.Dense(1, activation='relu')(self.x)
        self.model = tf.keras.Model(self.input, self.output)
        self.history = None

    def train(self, x, y, loss, optimizer, epochs):
        self.model.compile(loss=loss, optimizer=optimizer)
        self.history = self.model.fit(x=x, y=y, epochs=epochs)
        return self.history

    def predict(self, x):
        return self.model.predict(x)


# TODO: Allow different number of layers

class MLP(Network):
    def __init__(self):
        super(MLP, self).__init__()
        self.model = tf.keras.models.Sequential()
        self.model.add(tf.keras.layers.Flatten())
        self.model.add(tf.keras.layers.Dense(128, activation='elu'))
        self.model.add(tf.keras.layers.Dense(16, activation='relu'))
        self.model.add(tf.keras.layers.Dense(8, activation='elu'))
        self.model.add(tf.keras.layers.Dense(1, activation='relu'))
        self.history = None

    def train(self, x, y, loss, optimizer, epochs):
        a, b = x[0].shape
        self.model.build(input_shape=(None, a, b)) # Should I replace None w/ num_models?
        self.model.compile(loss=loss, optimizer=optimizer)
        self.history = self.model.fit(x=x, y=y, epochs=epochs)
        return self.history

    def predict(self, x):
        return self.model.predict(x)