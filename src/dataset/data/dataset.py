from .base_dataset import Dataset

import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import ast
from .utils import indexify, one_hotify, flatten, convert_shapes


class LayerWiseDataset(Dataset):
    def __init__(self, **kwargs):
        super().__init__()
        self.file_path = kwargs['file_path']
        self.subset = kwargs['subset']
        self.load()

    def load(self):
        raw_data = pd.read_csv(self.file_path)
        if self.subset == 'all':  # do nothing
            self.data = raw_data
        else:  # Options for subset are ['pretrained', 'simple'].
            self.data = raw_data.loc[raw_data['result.type'] == self.subset]

    def prepare(self, **kwargs):
        target_layer = kwargs['target_layer']
        layer_set = {'GlobalAveragePooling2D', 'ZeroPadding2D', 'BatchNormalization', 'AveragePooling2D', 'Dropout',
                     'DepthwiseConv2D', 'Multiply', 'ReLU', 'Flatten', 'MaxPooling2D', 'Add', 'Conv2D', 'Normalization',
                     'Reshape', 'SeparableConv2D', 'InputLayer', 'Concatenate', 'Dense', 'Activation', 'Rescaling'}
        dense_features = ["input_shape","output_shape","units"]
        conv_features = ["input_shape","output_shape","filters", "kernel_size", "stride"]
        pool_features = ["input_shape","output_shape","filters (default=1)", "pool_size", "stride"]
        # TODO: The power of kernel/pool size can be taken wrt the dim (2d/3d).
        x = []
        y = []
        for model_index, row in self.data.iterrows():
            model = tf.keras.models.model_from_json(row['result.model'])
            power_layerwise = ast.literal_eval(row["result.power_layerwise"])
            for layer, power in zip(model.layers, power_layerwise):
                if target_layer in layer.__class__.__name__.lower():
                    SUCCESS = False
                    layer_config = layer.get_config()
                    if "dense" in target_layer:
                        x.append([*[dim for dim in layer.input_shape if dim!=None], *[dim for dim in layer.output_shape if dim!=None], layer_config["units"]])
                        SUCCESS = True
                    elif "conv" in target_layer:
                        try:
                            x.append([*[dim for dim in layer.input_shape if dim!=None], *[dim for dim in layer.output_shape if dim!=None], layer_config["filters"], layer_config["kernel_size"][0], layer_config["strides"][0]])
                            SUCCESS = True
                        except: # Possibly depth-wise conv
                            x.append([*[dim for dim in layer.input_shape if dim!=None], *[dim for dim in layer.output_shape if dim!=None], layer.output_shape[-1], layer_config["kernel_size"][0], layer_config["strides"][0]])
                            SUCCESS = True
                    elif "pool" in target_layer:
                        try:
                            x.append([*[dim for dim in layer.input_shape if dim != None],
                                      *[dim for dim in layer.output_shape if dim != None], 1,
                                      layer_config["pool_size"][0], layer_config["strides"][0]])
                            SUCCESS = True
                        except: # Ignore
                            pass
                    if SUCCESS:
                        y.append(power)
        return np.array(x, dtype=np.uint16), np.array(y, dtype=float)

    def preprocessing(self, x, y):
        y = np.abs(y)  # Take abs due to issues with CodeCarbon
        y = (y - np.min(y)) / (np.max(y) - np.min(y))  # Normalize
        return x, y

def split(x, y, split_ratio, shuffle, seed):
    '''

    Args:
        mode: Either one of the following options: ['train', 'validation', 'test']
        validation_split_ratio: The size of the validation set relative to the training set.
        test_split_ratio: The size of the test set relative to the development (training + validation) set.
        shuffle: bool
        seed: int

    Returns:
        Target set.
            x, y

    '''
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=split_ratio, random_state=seed, shuffle=shuffle)
    return x_train, x_test, y_train, y_test


class ModelWiseDataset(Dataset):
    def __init__(self, file_path, subset, include_features=False, one_hot=True):
        super(ModelWiseDataset, self).__init__(file_path, subset)
        self.subset = subset
        self.file_path = file_path
        self.one_hot = one_hot
        self.include_features = include_features
        self.num_ops = None

    def load(self):
        raw_data = pd.read_csv(self.file_path)
        if self.subset == 'all':  # do nothing
            self.data = raw_data
        else:  # Options for subset are ['pretrained', 'simple'].
            self.data = raw_data.loc[raw_data['subset'] == self.subset]


    def prepare(self):
        dense_features = ["input_shape","output_shape","units"]
        conv_features = ["input_shape","output_shape","filters", "kernel_size", "stride"]
        pool_features = ["input_shape","output_shape","filters (default=1)", "pool_size", "stride"]
        # TODO: The power of kernel/pool size can be taken wrt the dim (2d/3d).
        x = []
        y = []
        if self.include_features:
            for model_index, row in self.data.iterrows():
                print(f"Processing Model {model_index+1}/{len(self.data)}")
                try:
                    model = tf.keras.models.model_from_json(row['result.model'])
                    power = float(row['result.power'])
                except:
                    print(f"Error: Model {row['result.name']} with _id {row['_id']} could not be imported.")
                    continue
                model_x = []
                for layer in model.layers:
                    layer_config = layer.get_config()
                    if "dense" in layer.__class__.__name__.lower():
                        model_x.append([*[dim for dim in layer.input_shape if dim!=None],
                                        *[dim for dim in layer.output_shape if dim!=None],
                                        layer_config["units"]])
                    elif "conv" in layer.__class__.__name__.lower():
                        try:
                            model_x.append([*[dim for dim in layer.input_shape if dim!=None],
                                            *[dim for dim in layer.output_shape if dim!=None],
                                            layer_config["filters"], layer_config["kernel_size"][0],
                                            layer_config["strides"][0]])
                        except: # Possibly depth-wise conv
                            model_x.append([*[dim for dim in layer.input_shape if dim!=None],
                                            *[dim for dim in layer.output_shape if dim!=None],
                                            layer.output_shape[-1], layer_config["kernel_size"][0],
                                            layer_config["strides"][0]])
                    elif "pool" in layer.__class__.__name__.lower():
                        try:
                            model_x.append([*[dim for dim in layer.input_shape if dim != None],
                                            *[dim for dim in layer.output_shape if dim != None], 1,
                                            layer_config["pool_size"][0], layer_config["strides"][0]])
                        except: # Ignore
                            pass
                x.append(model_x)
                y.append(power)
        else:
            for model_index, row in self.data.iterrows():
                print(f"Processing Model {model_index + 1}/{len(self.data)}")
                try:
                    model = tf.keras.models.model_from_json(row['result.model'])
                    power = float(row['result.power'])
                except:
                    print(f"Error: Model {row['result.name']} with _id {row['_id']} could not be imported.")
                    continue
                model_x = []
                for layer in model.layers:
                    #layer_config = layer.get_config()
                    model_x.append([layer.__class__.__name__, layer.input_shape, layer.output_shape])
                    """model_x.append([layer.__class__.__name__,
                                    *[dim for dim in layer.input_shape if dim != None],
                                    *[dim for dim in layer.output_shape if dim != None]])"""

                x.append(model_x)
                y.append(power)

        x, y = self.preprocessing(x, y)
        return np.array(x, dtype=np.uint16), np.array(y, dtype=float)

    def preprocessing(self, x, y):
        y = np.abs(y)
        y = (y - np.min(y)) / (np.max(y) - np.min(y))
        if self.one_hot:
            x = one_hotify(x)
        else:
            num_ops, x = indexify(x)
            self.num_ops = num_ops
        x = convert_shapes(x)
        x = flatten(x)
        x = tf.keras.preprocessing.sequence.pad_sequences(x, padding='post')
        return x, y