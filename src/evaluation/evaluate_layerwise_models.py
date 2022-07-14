from autogluon.tabular import TabularPredictor
from src.dataset.data.dataset import LayerWiseDatasetv2Large
import yaml
from pathlib import Path
import pandas as pd
from src.models.metrics import *

yaml_path = '/home/denizkorkmaz/PycharmProjects/TUM/SS22/green-ml-daml/src/run_config_layerwise.yaml'
config = yaml.safe_load(Path(yaml_path).read_text())

method_name = "CatBoost_BAG_L1" # Please enter the name of the model here that you want to use.

model_paths = {
    'dense': '/home/denizkorkmaz/PycharmProjects/TUM/SS22/green-ml-daml/src/models/agModels-greenML-07-12-2022--02:15:12-pretrained-dense',
    'conv': '/home/denizkorkmaz/PycharmProjects/TUM/SS22/green-ml-daml/src/models/agModels-greenML-07-12-2022--02:02:38-pretrained-conv',
    'pool': '/home/denizkorkmaz/PycharmProjects/TUM/SS22/green-ml-daml/src/models/agModels-greenML-07-12-2022--02:25:59-pretrained-pool'
} # TODO: Please add your models here manually.

features = {
    'dense': ["batch_size", "num_layer_total", "input_size", "output_size", "hidden_size"],
    'conv': ["batch_size", "num_layer_total", "input_size", "output_size", "filters", "kernel_size", "stride"],
    'pool': ["batch_size", "num_layer_total", "input_size", "output_size", "filters (default=1)", "pool_size", "stride"],
    'inputlayer': ["batch_size", "num_layer_total", "input_size"],
    'pad': ["batch_size", "num_layer_total", "input_size", "output_size", "padding"],
    'normalization': ["batch_size", "num_layer_total", "input_size", "output_size"],
    'activation': ["batch_size", "num_layer_total", "input_size"],
    'rescaling': ["batch_size", "num_layer_total", "input_size"],
    'reshape': ["batch_size", "num_layer_total", "input_size", "target_shape"],
    'dropout': ["batch_size", "num_layer_total", "input_size", "rate"],
    'add': ["batch_size", "num_layer_total", "output_size"],
    'multiply': ["batch_size", "num_layer_total", "output_size"],
    'concatenate': ["batch_size", "num_layer_total", "output_size"]
}

predictor_dense = TabularPredictor.load(model_paths['dense'])  # unnecessary, just demonstrates how to load previously-trained predictor from file
method_index = predictor_dense.get_model_names().index(method_name)
model_dense = predictor_dense.get_model_names()[method_index]

predictor_conv = TabularPredictor.load(model_paths['conv'])  # unnecessary, just demonstrates how to load previously-trained predictor from file
method_index = predictor_dense.get_model_names().index(method_name)
model_conv = predictor_conv.get_model_names()[method_index]

predictor_pool = TabularPredictor.load(model_paths['pool'])  # unnecessary, just demonstrates how to load previously-trained predictor from file
method_index = predictor_dense.get_model_names().index(method_name)
model_pool = predictor_pool.get_model_names()[method_index]

dataset_class_name = config['dataset_class']
dataset = LayerWiseDatasetv2Large(**config[dataset_class_name]['params'])
raw_x_test = dataset.raw_x_test
raw_y_test = dataset.raw_y_test
y_modelwise_true = dataset.y_modelwise_test
y_modelwise_predicted = list()
print(f"Number of Models in Test Set: {len(raw_x_test)}")

for i, model in enumerate(raw_x_test):
    print(f"Processing model {i+1}")
    current_consumption = 0.0
    for layer in model:
        layer_type = layer[0]
        layer_features = layer[1:]
        df = pd.DataFrame([layer_features], columns=features[layer_type])
        if layer_type == "dense":
            layerwise_prediction = predictor_dense.predict(df, model=model_dense).to_numpy().item()
            current_consumption += layerwise_prediction
        elif layer_type == "conv":
            layerwise_prediction = predictor_conv.predict(df, model=model_conv).to_numpy().item()
            current_consumption += layerwise_prediction
        elif layer_type == "pool":
            layerwise_prediction = predictor_pool.predict(df, model=model_pool).to_numpy().item()
            current_consumption += layerwise_prediction
        else:
            raise TypeError("Illegal layer type!")
    y_modelwise_predicted.append(current_consumption*1e-9)

print(len(y_modelwise_predicted))
print(len(y_modelwise_true))
print(y_modelwise_predicted)
print(y_modelwise_true)

my_metrics = {
    'mae': mae,
    'mse': mse,
    'rmse': rmse,
    'rmspe': rmspe,
    'r2': r2
}

loss_results = dict()
for metric in my_metrics:
    loss_fn = my_metrics[metric]
    loss_results[metric] = loss_fn(np.array(y_modelwise_predicted),np.array(y_modelwise_true))
print(loss_results)