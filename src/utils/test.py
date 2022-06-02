import energyEvaluation as eval
import tensorflow as tf

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, activation = "relu"),
    tf.keras.layers.Dense(10)
])

print(eval.evaluate_energy_forward(model, input_size = (10,), batch_size = 10, repetitions = 10))