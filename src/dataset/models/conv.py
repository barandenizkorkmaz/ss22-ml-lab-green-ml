import numpy as np
import tensorflow as tf


def conv_model(model_depth: int, build=True):
    model = tf.keras.Sequential()
    for depth in range(model_depth):
        chckr = np.random.randint(1, 5)
        if chckr != 4: # 3 times out of 4
            model.add(tf.keras.layers.Conv2D(np.random.randint(1, 5), 3, padding='same'))
        else:
            model.add(tf.keras.layers.MaxPooling2D())

    if build:
        model.build(input_shape=(1, 1024, 1024, 3))
    return model

def model_list(model_depths: list, build):
    return [conv_model(depth, build=build) for depth in model_depths]

