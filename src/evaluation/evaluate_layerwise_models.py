from autogluon.tabular import TabularPredictor
from src.dataset.data.dataset import LayerWiseDatasetv2Large
import yaml
from pathlib import Path
import pandas as pd
from src.models.metrics import *

yaml_path = '/home/denizkorkmaz/PycharmProjects/TUM/SS22/green-ml-daml/src/run_config_layerwise.yaml'
config = yaml.safe_load(Path(yaml_path).read_text())

method_name = "KNeighborsDist_BAG_L1" # Please enter the name of the model here that you want to use.

model_paths = {
    'dense': '/home/denizkorkmaz/PycharmProjects/TUM/SS22/green-ml-daml/src/models/agModels-greenML-07-14-2022--18:24:57-pretrained-dense',
    'conv': '/home/denizkorkmaz/PycharmProjects/TUM/SS22/green-ml-daml/src/models/agModels-greenML-07-14-2022--18:26:12-pretrained-conv',
    'pool': '/home/denizkorkmaz/PycharmProjects/TUM/SS22/green-ml-daml/src/models/agModels-greenML-07-14-2022--18:28:18-pretrained-pool',
    'inputlayer': '/home/denizkorkmaz/PycharmProjects/TUM/SS22/green-ml-daml/src/models/agModels-greenML-07-14-2022--18:28:55-pretrained-inputlayer',
    'pad': '/home/denizkorkmaz/PycharmProjects/TUM/SS22/green-ml-daml/src/models/agModels-greenML-07-14-2022--18:29:26-pretrained-pad',
    'normalization': '/home/denizkorkmaz/PycharmProjects/TUM/SS22/green-ml-daml/src/models/agModels-greenML-07-14-2022--18:30:32-pretrained-normalization',
    'activation': '/home/denizkorkmaz/PycharmProjects/TUM/SS22/green-ml-daml/src/models/agModels-greenML-07-14-2022--18:32:52-pretrained-activation',
    'rescaling': '/home/denizkorkmaz/PycharmProjects/TUM/SS22/green-ml-daml/src/models/agModels-greenML-07-14-2022--18:34:31-pretrained-rescaling',
    'reshape': '/home/denizkorkmaz/PycharmProjects/TUM/SS22/green-ml-daml/src/models/agModels-greenML-07-14-2022--18:35:01-pretrained-reshape',
    'dropout': '/home/denizkorkmaz/PycharmProjects/TUM/SS22/green-ml-daml/src/models/agModels-greenML-07-14-2022--18:36:16-pretrained-dropout',
    'add': '/home/denizkorkmaz/PycharmProjects/TUM/SS22/green-ml-daml/src/models/agModels-greenML-07-14-2022--18:37:26-pretrained-add',
    'multiply': '/home/denizkorkmaz/PycharmProjects/TUM/SS22/green-ml-daml/src/models/agModels-greenML-07-14-2022--18:38:57-pretrained-multiply',
    'concatenate': '/home/denizkorkmaz/PycharmProjects/TUM/SS22/green-ml-daml/src/models/agModels-greenML-07-14-2022--18:40:03-pretrained-concatenate'
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

predictor_dense = TabularPredictor.load(model_paths['dense'])
method_index = predictor_dense.get_model_names().index(method_name)
model_dense = predictor_dense.get_model_names()[method_index]

predictor_conv = TabularPredictor.load(model_paths['conv'])
method_index = predictor_conv.get_model_names().index(method_name)
model_conv = predictor_conv.get_model_names()[method_index]

predictor_pool = TabularPredictor.load(model_paths['pool'])
method_index = predictor_pool.get_model_names().index(method_name)
model_pool = predictor_pool.get_model_names()[method_index]

predictor_inputlayer = TabularPredictor.load(model_paths['inputlayer'])
method_index = predictor_inputlayer.get_model_names().index(method_name)
model_inputlayer = predictor_inputlayer.get_model_names()[method_index]

predictor_pad = TabularPredictor.load(model_paths['pad'])
method_index = predictor_pad.get_model_names().index(method_name)
model_pad = predictor_pad.get_model_names()[method_index]

predictor_normalization = TabularPredictor.load(model_paths['normalization'])
method_index = predictor_normalization.get_model_names().index(method_name)
model_normalization = predictor_normalization.get_model_names()[method_index]

predictor_activation = TabularPredictor.load(model_paths['activation'])
method_index = predictor_activation.get_model_names().index(method_name)
model_activation = predictor_activation.get_model_names()[method_index]

predictor_rescaling = TabularPredictor.load(model_paths['rescaling'])
method_index = predictor_rescaling.get_model_names().index(method_name)
model_rescaling = predictor_rescaling.get_model_names()[method_index]

predictor_reshape = TabularPredictor.load(model_paths['reshape'])
method_index = predictor_reshape.get_model_names().index(method_name)
model_reshape = predictor_reshape.get_model_names()[method_index]

predictor_dropout = TabularPredictor.load(model_paths['dropout'])
method_index = predictor_dropout.get_model_names().index(method_name)
model_dropout = predictor_dropout.get_model_names()[method_index]

predictor_add = TabularPredictor.load(model_paths['add'])
method_index = predictor_add.get_model_names().index(method_name)
model_add = predictor_add.get_model_names()[method_index]

predictor_multiply = TabularPredictor.load(model_paths['multiply'])
method_index = predictor_multiply.get_model_names().index(method_name)
model_multiply = predictor_multiply.get_model_names()[method_index]

predictor_concatenate = TabularPredictor.load(model_paths['concatenate'])
method_index = predictor_concatenate.get_model_names().index(method_name)
model_concatenate = predictor_concatenate.get_model_names()[method_index]


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
        elif layer_type == "inputlayer":
            layerwise_prediction = predictor_inputlayer.predict(df, model=model_inputlayer).to_numpy().item()
            current_consumption += layerwise_prediction
        elif layer_type == "pad":
            layerwise_prediction = predictor_pad.predict(df, model=model_pad).to_numpy().item()
            current_consumption += layerwise_prediction
        elif layer_type == "normalization":
            layerwise_prediction = predictor_normalization.predict(df, model=model_normalization).to_numpy().item()
            current_consumption += layerwise_prediction
        elif layer_type == "activation":
            layerwise_prediction = predictor_activation.predict(df, model=model_activation).to_numpy().item()
            current_consumption += layerwise_prediction
        elif layer_type == "rescaling":
            layerwise_prediction = predictor_rescaling.predict(df, model=model_rescaling).to_numpy().item()
            current_consumption += layerwise_prediction
        elif layer_type == "reshape":
            layerwise_prediction = predictor_reshape.predict(df, model=model_reshape).to_numpy().item()
            current_consumption += layerwise_prediction
        elif layer_type == "dropout":
            layerwise_prediction = predictor_dropout.predict(df, model=model_dropout).to_numpy().item()
            current_consumption += layerwise_prediction
        elif layer_type == "add":
            layerwise_prediction = predictor_add.predict(df, model=model_add).to_numpy().item()
            current_consumption += layerwise_prediction
        elif layer_type == "multiply":
            layerwise_prediction = predictor_multiply.predict(df, model=model_multiply).to_numpy().item()
            current_consumption += layerwise_prediction
        elif layer_type == "concatenate":
            layerwise_prediction = predictor_concatenate.predict(df, model=model_concatenate).to_numpy().item()
            current_consumption += layerwise_prediction
        else:
            raise TypeError("Illegal layer type!")
    current_consumption = current_consumption * 1e-9
    print(f"Predicted: {current_consumption}\tActual: {y_modelwise_true[i]}")
    y_modelwise_predicted.append(current_consumption)

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