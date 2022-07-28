# This script contains the model-wise methods.

import numpy as np
import tensorflow as tf
from .base_network import Network


# TODO: Add val dataset
class SimpleRNN(Network):
    def __init__(self, **kwargs):
        super(SimpleRNN, self).__init__()
        self.ragged = kwargs['ragged']
        self.epochs = kwargs['num_epochs']
        self.loss = kwargs['loss']
        self.lr = kwargs['lr']
        self.batch_size = kwargs['batch_size']
        self.model = tf.keras.models.Sequential()
        if kwargs['mask']:
            self.model.add(tf.keras.layers.Masking())
        self.model.add(tf.keras.layers.LSTM(kwargs['units'], return_sequences=False))
        self.model.add(tf.keras.layers.Dense(kwargs['hidden_dense'], activation='sigmoid'))
        self.model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
        self.history = None

    def train(self, x_train, y_train, x_val, y_val):
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)
        if not self.ragged:
            x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, padding='post')
            seq_len, feat_size = x_train[0].shape
        else:
            seq_len = None
            feat_size = x_train[0].shape[1]
        self.model.build(input_shape=(None, seq_len, feat_size)) # Should I replace None w/ num_models?
        self.model.compile(loss=self.loss, optimizer=optimizer)
        if x_val is not None and y_val is not None:
            self.history = self.model.fit(x=x_train, y=y_train,
                                          batch_size=self.batch_size,
                                          epochs=self.epochs,
                                          validation_data=(x_val, y_val))
        else:
            self.history = self.model.fit(x=x_train, y=y_train,
                                          batch_size=self.batch_size,
                                          epochs=self.epochs,
                                          validation_data=None)
        return self.history

    def predict(self, x_test):
        return self.model.predict(x_test)


class EmbeddedRNN(Network):
    def __init__(self, **kwargs):
        super(EmbeddedRNN, self).__init__()
        self.input = tf.keras.Input(shape=(kwargs['seq_len'], kwargs['feat_size']))
        self.epochs = kwargs['num_epochs']
        self.loss = kwargs['loss']
        self.lr = kwargs['lr']
        self.batch_size = kwargs['batch_size']
        self.x = tf.keras.layers.Embedding(kwargs['num_ops'], kwargs['out_dim'])(self.input[:, :, 0])
        self.x = tf.keras.layers.Concatenate()((self.x, self.input[:, :, 1:]))
        self.x = tf.keras.layers.LSTM(kwargs['units'], return_sequences=False,
                                      input_shape=(kwargs['seq_len'], kwargs['out_dim']+6))(self.x)
        self.x = tf.keras.layers.Dense(kwargs['hidden_dense'], activation='sigmoid')(self.x)
        self.output = tf.keras.layers.Dense(1, activation='sigmoid')(self.x)
        self.model = tf.keras.Model(self.input, self.output)
        self.history = None

    def train(self, x_train, y_train, x_val, y_val):
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)
        self.model.compile(loss=self.loss, optimizer=optimizer)
        if x_val is not None and y_val is not None:
            self.history = self.model.fit(x=x_train, y=y_train,
                                          batch_size=self.batch_size,
                                          epochs=self.epochs,
                                          validation_data=(x_val, y_val))
        else:
            self.history = self.model.fit(x=x_train, y=y_train,
                                          batch_size=self.batch_size,
                                          epochs=self.epochs,
                                          validation_data=None)
        return self.history

    def predict(self, x_test):
        return self.model.predict(x_test)


# TODO: Allow different number of layers

class MLP(Network):
    def __init__(self, **kwargs):
        super(MLP, self).__init__()
        self.epochs = kwargs['epochs']
        self.loss = kwargs['loss']
        self.lr = kwargs['lr']
        self.batch_size = kwargs['batch_size']
        self.model = tf.keras.models.Sequential()
        self.model.add(tf.keras.layers.Flatten())
        self.model.add(tf.keras.layers.Dense(128, activation=None))
        self.model.add(tf.keras.layers.PReLU())
        self.model.add(tf.keras.layers.Dense(64, activation=None))
        self.model.add(tf.keras.layers.PReLU())
        self.model.add(tf.keras.layers.Dense(32, activation=None))
        self.model.add(tf.keras.layers.PReLU())
        self.model.add(tf.keras.layers.Dense(16, activation=None))
        self.model.add(tf.keras.layers.PReLU())
        self.model.add(tf.keras.layers.Dense(8, activation=None))
        self.model.add(tf.keras.layers.PReLU())
        self.model.add(tf.keras.layers.Dense(1, activation=None))
        self.model.add(tf.keras.layers.PReLU())

        self.history = None

    def train(self, x_train, y_train, x_val, y_val):
        a, b = x_train[0].shape
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)
        self.model.build(input_shape=(None, a, b)) # Should I replace None w/ num_models?
        self.model.compile(loss=self.loss, optimizer=optimizer)
        self.history = self.model.fit(x=x_train, y=y_train,
                                      batch_size=self.batch_size,
                                      epochs=self.epochs,
                                      validation_data=(x_val, y_val))
        return self.history

    def predict(self, x):
        return self.model.predict(x)


class Transformer():
    def __init__(self, **kwargs):
        super(Transformer, self).__init__()
        self.epochs = kwargs['num_epochs']
        self.loss = kwargs['loss']
        self.lr = kwargs['lr']
        self.batch_size = kwargs['batch_size']
        self.input_shape = kwargs['input_shape']
        self.head_size = kwargs['head_size']
        self.num_heads = kwargs['num_heads']
        self.ff_dim = kwargs['ff_dim']
        self.num_transformer_blocks = kwargs['num_transformer_blocks']
        self.mlp_units = kwargs['mlp_units']
        self.dropout = kwargs['dropout'],
        self.mlp_dropout = kwargs['mlp_dropout'],
        self.inputs = tf.keras.Input(shape=self.input_shape)
        self.x = self.inputs
        for _ in range(self.num_transformer_blocks):
            self.x = self.transformer_encoder(inps=self.x,
                                              head_size=self.head_size,
                                              num_heads=self.num_heads,
                                              ff_dim=self.ff_dim,
                                              dropout=0.5)

        x = tf.keras.layers.GlobalAveragePooling1D(data_format="channels_first")(self.x)
        for dim in self.mlp_units:
            x = tf.keras.layers.Dense(dim, activation="sigmoid")(x)
            x = tf.keras.layers.Dropout(0.4)(x)
        outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
        self.model = tf.keras.Model(self.inputs, outputs)

    def transformer_encoder(self, inps, head_size, num_heads, ff_dim, dropout=0):
            # Normalization and Attention
            x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(inps)
            x = tf.keras.layers.MultiHeadAttention(
                key_dim=head_size, num_heads=num_heads, dropout=dropout
            )(x, x)
            x = tf.keras.layers.Dropout(dropout)(x)
            res = x + inps

            # Feed Forward Part
            x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(res)
            x = tf.keras.layers.Conv1D(filters=ff_dim, kernel_size=1, activation="sigmoid")(x)
            x = tf.keras.layers.Dropout(dropout)(x)
            x = tf.keras.layers.Conv1D(filters=inps.shape[-1], kernel_size=1)(x)
            return x + res

    def train(self, x_train, y_train, x_val, y_val):
        a, b = x_train[0].shape
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)
        self.model.build(input_shape=(None, a, b)) # Should I replace None w/ num_models?
        self.model.compile(loss=self.loss, optimizer=optimizer)
        tensorboard_callback = [tf.keras.callbacks.TensorBoard(log_dir=logdir),
                                tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)]

        if x_val is not None and y_val is not None:
            self.history = self.model.fit(x=x_train, y=y_train,
                                          batch_size=self.batch_size,
                                          epochs=self.epochs,
                                          validation_data=(x_val, y_val),
                                          callbacks=[tensorboard_callback])
        else:
            self.history = self.model.fit(x=x_train, y=y_train,
                                          batch_size=self.batch_size,
                                          epochs=self.epochs,
                                          validation_data=None)
        return self.history

    def predict(self, x_test):
        return self.model.predict(x_test)
