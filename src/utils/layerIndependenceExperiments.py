import tensorflow as tf
from tensorflow import keras

import evaluateLayers as eval

def simpleFullyConnected(numberLayers, hiddenSize):
    model = keras.Sequential()
    for i in range(numberLayers):
        model.add(keras.layers.Dense(hiddenSize, activation = "relu"))
    model.build((None,15))
    return model

modelDense = simpleFullyConnected(5, 20)

combined, individually = eval.evaluateLayers(modelDense)

print(f"Combined:{combined}, Individually: {sum(individually)} ")
eval.visualizeCombined(combined, individually, "Power consumption in kWh", "Combined power consumption vs individual layers", [f"layer {i}" for i in range(1,6)])

