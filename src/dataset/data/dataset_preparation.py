from base_dataset_preparation import DatasetPreparation

import os
import pandas as pd
import json
import ast
import shutil

class RegressorDatasetPreparation(DatasetPreparation):
    def __init__(self, path, dataset_name):
        super().__init__(path, dataset_name)
        self.raw_data = self.load()
        self.prepared_data = self.prepare()
        self.save()

    def load(self):
        return pd.read_csv(self.path)

    def prepare(self):
        layer_set = {'GlobalAveragePooling2D', 'ZeroPadding2D', 'BatchNormalization', 'AveragePooling2D', 'Dropout',
                     'DepthwiseConv2D', 'Multiply', 'ReLU', 'Flatten', 'MaxPooling2D', 'Add', 'Conv2D', 'Normalization',
                     'Reshape', 'SeparableConv2D', 'InputLayer', 'Concatenate', 'Dense', 'Activation', 'Rescaling'}
        prepared_data = {
            "dense": {},
            "conv": {}
        }
        prepared_data["dense"]["features"] = ["hidden_size"]
        prepared_data["conv"]["features"] = ["filter_size", "kernel_size", "stride"]
        for model_index, row in self.raw_data.iterrows():
            model_config = json.loads(row['result.model'])
            layers = model_config["config"]["layers"]
            power_layerwise = ast.literal_eval(row["result.power_layerwise"])
            for layer_index, (layer, power) in enumerate(zip(layers, power_layerwise)):
                layer_name = layer["class_name"].lower()
                layer_id = f"{model_index+1}-{layer_index+1}"
                layer_config = layer["config"]
                y = power
                if "dense" in layer_name:
                    x = [layer_config["units"]]
                    prepared_data["dense"][layer_id] = {
                        "x": x,
                        "y": y
                    }
                elif "conv" in layer_name:
                    try:
                        x = [layer_config["filters"], layer_config["kernel_size"][0], layer_config["strides"][0]]
                        prepared_data["conv"][layer_id] = {
                            "x": x,
                            "y": y
                        }
                    except: # Depth-wise Conv
                        pass
                else:
                    pass
        return prepared_data

    def save(self):
        root = os.path.join(os.getcwd(), self.dataset_name)
        if os.path.isdir(root):
            _input = input(f"Folder {root} already exists. Overwrite? [Y]es | [N]o\n").lower()
            if _input == 'n':
                raise SystemExit(0)
            else:
                shutil.rmtree(root)
        os.makedirs(root)

        for layer_type in self.prepared_data:
            features = self.prepared_data[layer_type]['features']
            layer_type_root = os.path.join(root,layer_type)
            for layer_id, layer_data in self.prepared_data[layer_type].items():
                if 'x' in layer_data and 'y' in layer_data:
                    layer_path = os.path.join(os.path.join(layer_type_root,layer_id))
                    os.makedirs(layer_path)
                    x = {
                        'features':features,
                        'x':layer_data['x']
                    }
                    y = {
                        'y':layer_data['y']
                    }
                    with open(f'{layer_path}/x.txt', 'w') as f:
                        f.write(json.dumps(x))
                    with open(f'{layer_path}/y.txt', 'w') as f:
                        f.write(json.dumps(y))

regressorDataset = RegressorDatasetPreparation(path="/home/denizkorkmaz/PycharmProjects/TUM/SS22/green-ml-daml/src/dataset/dataloader/dataset_layerwise.csv",
                                               dataset_name="regressor_dataset")