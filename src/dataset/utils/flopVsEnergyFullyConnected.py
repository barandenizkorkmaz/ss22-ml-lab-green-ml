from evaluation.energyEvaluation import evaluate_energy_forward
from evaluation.flopEvaluation import get_flops

import tensorflow as tf
from tensorflow import keras
from keras import layers


import matplotlib.pyplot as plt

def flopVEnergy(model, input_size):
    flops = get_flops(model)
    energy = evaluate_energy_forward(model, input_size, batch_size = 50, repetitions = 1000)

    return (flops, energy)

def visualizePairs(pairs, xlabel, ylabel, title, labels= None):
    x,y = zip(*pairs)
    fig = plt.figure()
    ax1 = fig.add_subplot()
    ax1.scatter(x,y)
    if labels:
        for i, txt in enumerate(labels):
            ax1.annotate(txt, (x[i],y[i]))
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    ax1.set_title(title)
    plt.show()

def simpleFullyConnected(numberLayers, hiddenSize):
    model = keras.Sequential()
    for i in range(numberLayers):
        model.add(layers.Dense(hiddenSize, activation = "relu"))
    model.build((50,15))
    return model

def compareSimpleFullyConnected():
    models = [simpleFullyConnected(i, 100) for i in range(2,10)]
    results = [flopVEnergy(model, (15,)) for model in models]
    labels = [f"{i} layers" for i in range(2,10)]
    visualizePairs(results, xlabel = "Number of FLOPS", ylabel = "Power in kWh", title = "Power consumption vs. FLOPS for fully connected NNs", labels= labels)

def compareHiddenSize():
    hiddenSizes = [i for i in range(10, 100, 10)]
    models = [simpleFullyConnected(5, i) for i in hiddenSizes]
    results = [flopVEnergy(model, (15,)) for model in models]
    labels = [f"size of layer: {i}" for i in hiddenSizes]
    visualizePairs(results, xlabel = "Number of FLOPS", ylabel = "Power in kWh", title = "Power consumption vs. FLOPS for fully connected NNs", labels= labels)

compareHiddenSize()
#compareSimpleFullyConnected()