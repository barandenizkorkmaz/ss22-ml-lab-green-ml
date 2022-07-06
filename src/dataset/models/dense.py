import tensorflow as tf
from tensorflow import keras

def fully_connected(number_layers, hidden_size, input_size):
    model = keras.Sequential()
    for i in range(number_layers):
        model.add(keras.layers.Dense(hidden_size, activation = "relu"))
    model.build((None,input_size))
    return model

def get_model(params):
    model = fully_connected(params["number_layers"], params["hidden_size"], params["hidden_size"])
    return model, f"Dense {params['number_layers']}-layers with hidden size of {params['hidden_size']} and input size of {params['hidden_size']}"