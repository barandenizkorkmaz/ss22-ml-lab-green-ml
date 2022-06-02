import tensorflow as tf
from tensorflow import keras

def fully_connected(number_layers, hidden_size):
    model = keras.Sequential()
    for i in range(number_layers):
        model.add(keras.layers.Dense(hidden_size, activation = "relu"))
    model.build((None,hidden_size))
    return model

def get_model(number_layers: int):
    model = fully_connected(number_layers, 20)
    return model, f"Dense {number_layers}-layers"