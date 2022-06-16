import numpy as np
import tensorflow as tf


def conv_model(number_layers: int):
    model = tf.keras.Sequential()
    for _ in range(number_layers):
        chckr = np.random.randint(1, 5)
        if chckr != 4: # 3 times out of 4
            model.add(tf.keras.layers.Conv2D(np.random.randint(1, 5), 3, padding='same'))
        else:
            model.add(tf.keras.layers.MaxPooling2D())
    model.build(input_shape=(None, 1024, 1024, 3))
    return model

def get_model(params):
    model = conv_model(params['number_layers'])
    return model, f"Conv with {params['number_layers']} number of layers"