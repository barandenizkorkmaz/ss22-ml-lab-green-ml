#!pip install codecarbon

from codecarbon import EmissionsTracker
import tensorflow as tf
from tensorflow.keras.applications import *

from PIL import Image
import numpy as np

print(tf.__version__)
physical_devices = tf.config.experimental.list_physical_devices('GPU')
print(physical_devices)
if physical_devices:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)

def standardize(image_data):
    image_data -= np.mean(image_data, axis=0)
    image_data /= np.std(image_data, axis=0)
    return image_data

img = Image.open("../../img/dog.jpg")
img  = tf.keras.preprocessing.image.img_to_array(img)

print(f"Shape of Image Before Preprocessing: {img.shape}")

resizing_layer = tf.keras.layers.Resizing(256,256)
center_crop_layer = tf.keras.layers.CenterCrop(224,224)
preprocessed_img = resizing_layer(img)
preprocessed_img = center_crop_layer(preprocessed_img)
preprocessed_img = standardize(preprocessed_img)
img = np.expand_dims(preprocessed_img, axis=0)

print(f"Shape of Image After Preprocessing: {img.shape}")

models = {
    'vgg16': VGG16(),
    'vgg19': VGG19(),
    'mobilenet': MobileNet(),
    'mobilenetv2': MobileNetV2(),
    'mobilenetv3small': MobileNetV3Small(),
    'mobilenetv3large': MobileNetV3Large(),
    'densenet121': DenseNet121(),
    'densenet169':DenseNet169(),
    'densenet201':DenseNet201(),
    'resnet50v2':ResNet50V2(),
    'resnet101v2':ResNet101V2(),
    'resnet152v2':ResNet152V2()
}

import pandas as pd
file = 'emissions.csv'
path = '/content/drive/My Drive/emissions.csv'

NUM_FORWARD_PASS = 50000
FREQUENCY = 1000
for model_id, model_name in enumerate(models):
  tracker = EmissionsTracker(project_name=model_name)
  tracker.start()
  model = models[model_name]
  # GPU intensive training code
  for i in range(NUM_FORWARD_PASS):
    if ((i+1) % FREQUENCY) == 0:
      print(f"Model Id: {model_id + 1}/{len(models)}\tModel: {model_name}\tForward Pass: {i+1}/{NUM_FORWARD_PASS} finished!")
    output = model.predict(img)
  emissions = tracker.stop()
  df = pd.read_csv(file)
  with open(path, 'w', encoding = 'utf-8-sig') as f:
    df.to_csv(f)