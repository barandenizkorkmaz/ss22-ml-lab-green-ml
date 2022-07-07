import tensorflow as tf
from tensorflow.keras.applications import *
from tensorflow.keras.layers import *

models = [
    'DenseNet121',
    'DenseNet169',
    'EfficientNetB0',
    'EfficientNetB1',
    'EfficientNetB2',
    'EfficientNetB3',
    'EfficientNetB4',
    'EfficientNetV2B0',
    'EfficientNetV2B1',
    'EfficientNetV2B2',
    'EfficientNetV2S',
    'InceptionV3',
    'MobileNet',
    'MobileNetV2',
    'ResNet101',
    'ResNet101V2',
    'ResNet152',
    'ResNet152V2',
    'ResNet50',
    'ResNet50V2',
    'VGG16',
    'VGG19',
    'Xception'
]

def get_base_model(model_name, input_shape):
    base_models = {
        'DenseNet121': DenseNet121(input_shape=input_shape, include_top=False, weights=None),
        'DenseNet169': DenseNet169(input_shape=input_shape, include_top=False, weights=None),
        'EfficientNetB0': EfficientNetB0(input_shape=input_shape, include_top=False, weights=None),
        'EfficientNetB1': EfficientNetB1(input_shape=input_shape, include_top=False, weights=None),
        'EfficientNetB2': EfficientNetB2(input_shape=input_shape, include_top=False, weights=None),
        'EfficientNetB3': EfficientNetB3(input_shape=input_shape, include_top=False, weights=None),
        'EfficientNetB4': EfficientNetB4(input_shape=input_shape, include_top=False, weights=None),
        'EfficientNetV2B0': EfficientNetV2B0(input_shape=input_shape, include_top=False, weights=None),
        'EfficientNetV2B1': EfficientNetV2B1(input_shape=input_shape, include_top=False, weights=None),
        'EfficientNetV2B2': EfficientNetV2B2(input_shape=input_shape, include_top=False, weights=None),
        'EfficientNetV2S': EfficientNetV2S(input_shape=input_shape, include_top=False, weights=None),
        'InceptionV3': InceptionV3(input_shape=input_shape, include_top=False, weights=None),
        'MobileNet': MobileNet(input_shape=input_shape, include_top=False, weights=None),
        'MobileNetV2': MobileNetV2(input_shape=input_shape, include_top=False, weights=None),
        'ResNet101': ResNet101(input_shape=input_shape, include_top=False, weights=None),
        'ResNet101V2': ResNet101V2(input_shape=input_shape, include_top=False, weights=None),
        'ResNet152': ResNet152(input_shape=input_shape, include_top=False, weights=None),
        'ResNet152V2': ResNet152V2(input_shape=input_shape, include_top=False, weights=None),
        'ResNet50': ResNet50(input_shape=input_shape, include_top=False, weights=None),
        'ResNet50V2': ResNet50V2(input_shape=input_shape, include_top=False, weights=None),
        'VGG16': VGG16(input_shape=input_shape, include_top=False, weights=None),
        'VGG19': VGG19(input_shape=input_shape, include_top=False, weights=None),
        'Xception': Xception(input_shape=input_shape, include_top=False, weights=None)
    }
    return base_models[model_name]

def get_model(params):
    model_name = models[params['model_index']]
    input_size = params['input_size']
    model = tf.keras.Sequential(
        [
            get_base_model(model_name, (input_size,input_size,3)),
            GlobalAveragePooling2D(),
            Dense(4096),
            Dense(2048),
            Dense(1000),
            Dense(512),
            Dense(256),
            Dense(100),
            Dense(50),
            Dense(25),
            Dense(10)
        ]
    )
    return model, model_name