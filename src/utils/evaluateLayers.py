from energyEvaluation import evaluate_energy_forward
import tensorflow as tf
import matplotlib.pyplot as plt

def evaluateLayers(model):
    input_size = model.input_shape[1:]
    resultCombined = evaluate_energy_forward(model, input_size= input_size, batch_size=20, repetitions = 100)
    resultsIndividually = []
    for layer in model.layers:
        input_size = layer.input_shape[1:] #first parameter is batch_size
        resultsIndividually.append(evaluate_energy_forward(layer, input_size = input_size, batch_size= 20, repetitions = 100))

    return resultCombined, resultsIndividually

def visualizeCombined(combined, individually, ylabel, title, labels):

    fig = plt.figure()
    ax1 = fig.add_subplot()
    ax1.bar([0],[combined], width = 0.5)
    for i in range(len(individually)):
        p = ax1.bar([1], [individually[i]], width = 0.5, bottom = sum(individually[:i]))
        ax1.bar_label(p,labels = [labels[i]], label_type = "center")
    
    ax1.set_ylabel(ylabel)
    ax1.set_xticks([0,1], labels= ["combined", "individually"])
    ax1.set_title(title)
    plt.show()

#This is ugly at the moment... Should be in a different file
model1 = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, activation = "relu", input_shape = (20,)),
    tf.keras.layers.Dense(10)
])
model2 = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(128, input_shape = (10,8)),
    tf.keras.layers.Dense(10)
])

model3 = tf.keras.models.Sequential([
    tf.keras.layers.GRU(256, return_sequences= True, input_shape = (10,8)),
    tf.keras.layers.SimpleRNN(128),
    tf.keras.layers.Dense(10)
])

model4 = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64,3, input_shape = (224,224,3)),
    tf.keras.layers.Conv2D(64,3),
    tf.keras.layers.MaxPooling2D(pool_size = (2,2), strides = 2),
    tf.keras.layers.Conv2D(128,3),
    tf.keras.layers.Conv2D(128,3),
    tf.keras.layers.MaxPooling2D(pool_size = (2,2), strides = 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(20)
])





#model2.summary()

#visualizeCombined(*evaluateLayers(model2), "Power consumption in kWh", "Combined power consumption vs individual layers", ["LSTM", "Dense"])

#visualizeCombined(*evaluateLayers(model3), "Power consumption in kWh", "Combined power consumption vs individual layers", ["GRU", "LSTM", "Dense"])

#visualizeCombined(*evaluateLayers(model4), "Power consumption in kWh", "Combined power consumption vs individual layers", ["Conv64", "Conv64", "MaxPool", "Conv128", "Conv128", "MaxPool", "Flatten", "Dense"])