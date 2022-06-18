import json

from src.dataset.data.dataset import LayerWiseDataset
from flopEvaluation import get_flops

import tensorflow as tf
import keras
import matplotlib.pyplot as plt
import numpy as np
import ast

dataset_path = "/home/denizkorkmaz/PycharmProjects/TUM/SS22/green-ml-daml/src/dataset/dataloader/dataset_layerwise.csv"
target_model = "VGG16"

layerWiseDataset = LayerWiseDataset(dataset_path)

x = []
y = []

for model_index, row in layerWiseDataset.raw_data.iterrows():
    model_name = row['result.name']
    if model_name != target_model:
        continue
    model = tf.keras.models.model_from_json(row['result.model'])
    power_layerwise = ast.literal_eval(row["result.power_layerwise"])
    for layer, power in zip(model.layers, power_layerwise):
        layer_name = layer.__class__.__name__
        print(f"Layer: {layer_name}\tPower: {power}")
        x.append(layer_name)
        y.append(power)

tick_locations = np.arange(len(x))  # the label locations
width = 0.35  # the width of the bars

ROUND = 1
y = [round(elem*1e9,ROUND) for elem in y]

fig, ax = plt.subplots()
rects1 = ax.bar(tick_locations, y, width, label=target_model)

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Avg. Energy Consumption (nW)')
ax.set_title(f'{target_model} Layer-Wise Energy Consumption')
ax.set_xticks(tick_locations, x, rotation=90)
ax.legend()

ax.bar_label(rects1, padding=3)

fig.tight_layout()
plt.savefig(f"{target_model}-layerwise-energy-consumption",dpi=1200)
plt.show()