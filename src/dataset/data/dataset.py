from .base_dataset import Dataset

import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import ast

class LayerWiseDataset(Dataset):
    def __init__(self, file_path, subset):
        super().__init__(file_path, subset)

    def load(self):
        raw_data = pd.read_csv(self.file_path)
        if self.subset == 'all':  # do nothing
            self.data = raw_data
        else:  # Options for subset are ['pretrained', 'simple'].
            self.data = raw_data.loc[raw_data['result.type'] == self.subset]

    def prepare(self, target_layer):
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