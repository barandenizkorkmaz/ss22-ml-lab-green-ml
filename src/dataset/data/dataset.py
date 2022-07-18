from .base_dataset import Dataset

import os
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import ast
from .utils import indexify, one_hotify, flatten, convert_shapes, extract_layer_features

class LayerWiseDatasetv2Small(Dataset):
    def __init__(self, **kwargs):
        super().__init__()
        self.dataset_class = 'LayerWiseDatasetv2Small'
        self.file_path = kwargs['file_path']
        self.subset = kwargs['subset']
        self.features = {
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
        layer_set = {'GlobalAveragePooling2D', 'ZeroPadding2D', 'BatchNormalization', 'AveragePooling2D', 'Dropout',
                     'DepthwiseConv2D', 'Multiply', 'ReLU', 'Flatten', 'MaxPooling2D', 'Add', 'Conv2D', 'Normalization',
                     'Reshape', 'SeparableConv2D', 'InputLayer', 'Concatenate', 'Dense', 'Activation', 'Rescaling'}

        self.load_dataset(**kwargs)
        self.get_raw_splits(**kwargs)
        del self.raw_x, self.raw_y, self.y_modelwise
        self.x_train, self.y_train = self.raw_split_to_training_split(self.raw_x_train, self.raw_y_train,**kwargs)
        del self.raw_x_train, self.raw_y_train, self.y_modelwise_train
        self.x_test, self.y_test = self.raw_split_to_training_split(self.raw_x_test, self.raw_y_test,**kwargs)
        if kwargs['validation_split']:
            self.x_validation, self.y_validation = self.raw_split_to_training_split(self.raw_x_val, self.raw_y_val,**kwargs)
            del self.raw_x_val, self.raw_y_val, self.y_modelwise_val
        if kwargs['save_splits']:
            self.save_splits(**kwargs)

    def load_dataset(self, **kwargs):
        if kwargs['load_dataset'] and (os.path.isfile(kwargs['dataset_path_raw_x']) and os.path.isfile(kwargs['dataset_path_raw_y']) and os.path.isfile(kwargs['dataset_path_y_modelwise'])):
            self.load(**kwargs)
            print("Load_dataset successful!")
        else:
            print("Load_dataset failed. Preparing the dataset from the scratch.")
            self.load_csv()
            self.raw_x, self.raw_y, self.y_modelwise = self.get_raw_dataset(**kwargs)
            del self.data
            self.save()

    def load_csv(self):
        raw_data = pd.read_csv(self.file_path)
        if self.subset == 'all':  # do nothing
            self.data = raw_data
        else:  # Options for subset are ['pretrained', 'simple'].
            self.data = raw_data.loc[raw_data['result.type'] == self.subset]

    def get_raw_dataset(self, **kwargs):
        x = []
        y_layerwise = []
        y_modelwise = []
        for model_index, row in self.data.iterrows():
            print(f"Processing model {model_index}")
            current_model = []
            current_y_layerwise = []
            model = tf.keras.models.model_from_json(row['result.model'])
            num_layer_total = len(model.layers)
            power_model = row["result.power"]
            power_layerwise = ast.literal_eval(row["result.power_layerwise"])
            for layer, power_layer in zip(model.layers, power_layerwise):
                extract_layer_features_return = extract_layer_features(layer)
                if extract_layer_features_return != False:
                    layer_features, layer_type = extract_layer_features_return
                    if "config.batch_size" in self.data:
                        batch_size = row["config.batch_size"]
                        layer_features = [layer_type, batch_size, num_layer_total, *layer_features]
                    else:
                        layer_features = [layer_type, num_layer_total, *layer_features]
                    current_model.append(layer_features)
                    current_y_layerwise.append(power_layer)
            if len(current_model) > 0:  # If any match for the corresponding layer is found on the current model
                x.append(current_model)
                y_modelwise.append(power_model)
                y_layerwise.append(current_y_layerwise)
        print(f"(getRawDataset)\tNumber of Models in the Dataset: {len(x)}")
        return x, y_layerwise, y_modelwise

    def get_raw_splits(self, **kwargs):
        x, y, y_modelwise = shuffle_iterables(True, 123, self.raw_x, self.raw_y, self.y_modelwise)
        self.raw_x_train, self.raw_x_test = split_2(x, split_ratio=kwargs['test_split'])
        self.raw_y_train, self.raw_y_test = split_2(y, split_ratio=kwargs['test_split'])
        self.y_modelwise_train, self.y_modelwise_test = split_2(y_modelwise, split_ratio=kwargs['test_split'])
        if kwargs['validation_split']:
            self.raw_x_train, self.raw_x_val = split_2(self.raw_x_train, split_ratio=kwargs['validation_split'])
            self.raw_y_train, self.raw_y_val = split_2(self.raw_y_train, split_ratio=kwargs['validation_split'])
            self.y_modelwise_train, self.y_modelwise_val = split_2(self.y_modelwise_train, split_ratio=kwargs['validation_split'])
        else:
            self.raw_x_val, self.raw_y_val, self.y_modelwise_val = None, None, None

    def raw_split_to_training_split(self, raw_x_split, raw_y_split, **kwargs):
        target_layer = kwargs['target_layer']
        tmp_x = []
        tmp_y = []
        for model_x, model_y in zip(raw_x_split, raw_y_split):
            for layer_x, layer_y in zip(model_x, model_y):
                if layer_x[0] == target_layer:
                    tmp_x.append(layer_x[1:])
                    tmp_y.append(layer_y)
        tmp_x = np.array(tmp_x, dtype=np.uint16)
        tmp_y = np.array(tmp_y, dtype=np.float)
        # Data Preprocessing
        tmp_x, tmp_y = self.preprocessing(tmp_x, tmp_y)
        return tmp_x, tmp_y

    def preprocessing(self, x, y):
        if type(y) is list:
            y = [tmp*1e9 for tmp in y]
        else:
            y = y * 1e9
        #y = np.abs(y)  # Take abs due to issues with CodeCarbon
        #y = (y - np.min(y)) / (np.max(y) - np.min(y))  # Normalize
        return x, y

    def load(self, *args, **kwargs):
        import pickle
        dataset_path_raw_x = kwargs['dataset_path_raw_x']
        dataset_path_raw_y = kwargs['dataset_path_raw_y']
        dataset_path_y_modelwise = kwargs['dataset_path_y_modelwise']
        with open(dataset_path_raw_x, 'rb') as f:
            self.raw_x = pickle.load(f)
        with open(dataset_path_raw_y, 'rb') as f:
            self.raw_y = pickle.load(f)
        with open(dataset_path_y_modelwise, 'rb') as f:
            self.y_modelwise = pickle.load(f)

    def save(self):
        """
        Saves the entire dataset. i.e. x and y
        """
        import pickle
        with open(f'{self.dataset_class}-{self.subset}-raw_x.pkl', 'wb') as f:
            pickle.dump(self.raw_x, f)
        with open(f'{self.dataset_class}-{self.subset}-raw_y.pkl', 'wb') as f:
            pickle.dump(self.raw_y, f)
        with open(f'{self.dataset_class}-{self.subset}-y_modelwise.pkl', 'wb') as f:
            pickle.dump(self.y_modelwise, f)

    def save_splits(self, *args, **kwargs):
        splits = {
            'train': (self.x_train, self.y_train),
            'test': (self.x_test, self.y_test)
        }
        if kwargs['validation_split'] != False:
            splits['validation'] = (self.x_validation, self.y_validation)
        for split in splits:
            x_split, y_split = splits[split]
            df = pd.DataFrame(x_split, columns = self.features[kwargs['target_layer']])
            df['energy_consumption'] = y_split
            df.to_csv(f"{self.dataset_class}-{self.subset}-{kwargs['target_layer']}-{split}.csv")

class LayerWiseDatasetv2Large(Dataset):
    def __init__(self, **kwargs):
        super().__init__()
        self.dataset_class = 'LayerWiseDatasetv2Large'
        self.file_path = kwargs['file_path']
        self.subset = kwargs['subset']
        self.features = {
            'dense': ["batch_size", "num_layer_total", "input_size", "output_size", "hidden_size"],
            'conv': ["batch_size", "num_layer_total", "input_size", "output_size", "filters", "kernel_size", "stride"],
            'pool': ["batch_size", "num_layer_total", "input_size", "output_size", "filters (default=1)", "pool_size",
                     "stride"],
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
        layer_set = {'GlobalAveragePooling2D', 'ZeroPadding2D', 'BatchNormalization', 'AveragePooling2D', 'Dropout',
                     'DepthwiseConv2D', 'Multiply', 'ReLU', 'Flatten', 'MaxPooling2D', 'Add', 'Conv2D', 'Normalization',
                     'Reshape', 'SeparableConv2D', 'InputLayer', 'Concatenate', 'Dense', 'Activation', 'Rescaling'}
        self.load_dataset(**kwargs)
        self.get_raw_splits(**kwargs)
        del self.raw_x, self.raw_y, self.y_modelwise
        self.x_train, self.y_train = self.raw_split_to_training_split(self.raw_x_train, self.raw_y_train,**kwargs)
        del self.raw_x_train, self.raw_y_train, self.y_modelwise_train
        self.x_test, self.y_test = self.raw_split_to_training_split(self.raw_x_test, self.raw_y_test,**kwargs)
        if kwargs['validation_split']:
            self.x_validation, self.y_validation = self.raw_split_to_training_split(self.raw_x_val, self.raw_y_val,**kwargs)
            del self.raw_x_val, self.raw_y_val, self.y_modelwise_val
        if kwargs['save_splits']:
            self.save_splits(**kwargs)

    def load_dataset(self, **kwargs):
        if kwargs['load_dataset'] and (os.path.isfile(kwargs['dataset_path_raw_x']) and os.path.isfile(kwargs['dataset_path_raw_y']) and os.path.isfile(kwargs['dataset_path_y_modelwise'])):
            self.load(**kwargs)
            print("Load_dataset successful!")
        else:
            print("Load_dataset failed. Preparing the dataset from the scratch.")
            self.load_csv()
            self.raw_x, self.raw_y, self.y_modelwise = self.get_raw_dataset(**kwargs)
            del self.data
            self.save()

    def load_csv(self):
        raw_data = pd.read_csv(self.file_path)
        if self.subset == 'all':  # do nothing
            self.data = raw_data
        else:  # Options for subset are ['pretrained', 'simple'].
            self.data = raw_data.loc[raw_data['result.type'] == self.subset]

    def get_raw_dataset(self, **kwargs):
        x = []
        y_layerwise = []
        y_modelwise = []
        for model_index, row in self.data.iterrows():
            current_model = []
            current_y_layerwise = []
            model = tf.keras.models.model_from_json(row['result.model'])
            power_model = row["result.power"]
            power_layerwise = ast.literal_eval(row["result.power_layerwise"])
            num_layer_base_model = len(model.layers[0].layers)
            num_layer_total = len(model.layers) + num_layer_base_model - 1
            print(f"(get_raw_dataset)\tProcessing model {model_index}\tModel Name: {row['result.name']}\tNum Layers in Base Model: {num_layer_base_model}")
            for i, layer in enumerate(model.layers):
                if i == 0:
                    for j, sublayer in enumerate(layer.layers):
                        layer_num = j
                        extract_layer_features_return = extract_layer_features(sublayer)
                        if extract_layer_features_return != False:
                            layer_features, layer_type = extract_layer_features_return
                            batch_size = row["config.batch_size"]
                            layer_features = [layer_type, batch_size, num_layer_total, *layer_features]
                            current_model.append(layer_features)
                            current_y_layerwise.append(power_layerwise[layer_num])
                else:
                    layer_num = num_layer_base_model + i - 1
                    extract_layer_features_return = extract_layer_features(layer)
                    if extract_layer_features_return != False:
                        layer_features, layer_type = extract_layer_features_return
                        batch_size = row["config.batch_size"]
                        layer_features = [layer_type, batch_size, num_layer_total, *layer_features]
                        current_model.append(layer_features)
                        current_y_layerwise.append(power_layerwise[layer_num])
            if len(current_model) > 0:  # If any match for the corresponding layer is found on the current model
                x.append(current_model)
                y_modelwise.append(power_model)
                y_layerwise.append(current_y_layerwise)
        print(f"(getRawDataset)\tNumber of Models in the Dataset: {len(x)}")
        return x, y_layerwise, y_modelwise

    def get_raw_splits(self, **kwargs):
        x, y, y_modelwise = shuffle_iterables(True, 123, self.raw_x, self.raw_y, self.y_modelwise)
        self.raw_x_train, self.raw_x_test = split_2(x, split_ratio=kwargs['test_split'])
        self.raw_y_train, self.raw_y_test = split_2(y, split_ratio=kwargs['test_split'])
        self.y_modelwise_train, self.y_modelwise_test = split_2(y_modelwise, split_ratio=kwargs['test_split'])
        if kwargs['validation_split']:
            self.raw_x_train, self.raw_x_val = split_2(self.raw_x_train, split_ratio=kwargs['validation_split'])
            self.raw_y_train, self.raw_y_val = split_2(self.raw_y_train, split_ratio=kwargs['validation_split'])
            self.y_modelwise_train, self.y_modelwise_val = split_2(self.y_modelwise_train, split_ratio=kwargs['validation_split'])
        else:
            self.raw_x_val, self.raw_y_val, self.y_modelwise_val = None, None, None

    def raw_split_to_training_split(self, raw_x_split, raw_y_split, **kwargs):
        target_layer = kwargs['target_layer']
        tmp_x = []
        tmp_y = []
        for model_x, model_y in zip(raw_x_split, raw_y_split):
            for layer_x, layer_y in zip(model_x, model_y):
                if layer_x[0] == target_layer:
                    tmp_x.append(layer_x[1:])
                    tmp_y.append(layer_y)
        tmp_x = np.array(tmp_x, dtype=np.int)
        tmp_y = np.array(tmp_y, dtype=np.float)
        # Data Preprocessing
        tmp_x, tmp_y = self.preprocessing(tmp_x, tmp_y)
        return tmp_x, tmp_y

    def preprocessing(self, x, y):
        y = y * 1e9
        #y = np.abs(y)  # Take abs due to issues with CodeCarbon
        #y = (y - np.min(y)) / (np.max(y) - np.min(y))  # Normalize
        return x, y

    def load(self, *args, **kwargs):
        import pickle
        dataset_path_raw_x = kwargs['dataset_path_raw_x']
        dataset_path_raw_y = kwargs['dataset_path_raw_y']
        dataset_path_y_modelwise = kwargs['dataset_path_y_modelwise']
        with open(dataset_path_raw_x, 'rb') as f:
            self.raw_x = pickle.load(f)
        with open(dataset_path_raw_y, 'rb') as f:
            self.raw_y = pickle.load(f)
        with open(dataset_path_y_modelwise, 'rb') as f:
            self.y_modelwise = pickle.load(f)

    def save(self):
        """
        Saves the entire dataset. i.e. x and y
        """
        import pickle
        with open(f'{self.dataset_class}-{self.subset}-raw_x.pkl', 'wb') as f:
            pickle.dump(self.raw_x, f)
        with open(f'{self.dataset_class}-{self.subset}-raw_y.pkl', 'wb') as f:
            pickle.dump(self.raw_y, f)
        with open(f'{self.dataset_class}-{self.subset}-y_modelwise.pkl', 'wb') as f:
            pickle.dump(self.y_modelwise, f)

    def save_splits(self, *args, **kwargs):
        splits = {
            'train': (self.x_train, self.y_train),
            'test': (self.x_test, self.y_test)
        }
        if kwargs['validation_split'] != False:
            splits['validation'] = (self.x_validation, self.y_validation)
        for split in splits:
            x_split, y_split = splits[split]
            df = pd.DataFrame(x_split, columns = self.features[kwargs['target_layer']])
            df['energy_consumption'] = y_split
            df.to_csv(f"{self.dataset_class}-{self.subset}-{kwargs['target_layer']}-{split}.csv")

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


def split_2(x, split_ratio):
    """
    Function for manually splitting a list by a given split_ratio
    Args:
        x: list
        split_ratio: float

    Returns:
        split_1, split_2: list, list

    """
    split_point = int(len(x) * (1 - split_ratio))
    return x[:split_point], x[split_point:]

def shuffle_iterables(shuffle, seed, *args):
    """
    Shuffles any number of lists with each indices connected (corresponding to each other).

    Args:
        shuffle: bool
        seed: int
        *args: any number of lists that you want to shuffle simultaneously

    Returns:
        tuple of lists in the same order

    """
    it = iter(args)
    the_len = len(next(it))
    if not all(len(l) == the_len for l in it):
        raise ValueError('not all lists have same length!')
    if not shuffle:
        return args
    else:
        iters = list()
        for _ in range(len(args)):
            iters.append(list())
        import random
        random.seed(seed)
        shuffled_order = list(range(len(args[0])))
        random.shuffle(shuffled_order)
        for index in shuffled_order:
            for i, iterable in enumerate(iters):
                iters[i].append(args[i][index])
        return tuple(iters)

class ModelWiseDataset(Dataset):
    def __init__(self, **kwargs):
        super(ModelWiseDataset, self).__init__(kwargs['file_path'], kwargs['subset'])
        self.subset = kwargs['subset']
        self.file_path = kwargs['file_path']
        self.one_hot = kwargs['one_hot']
        self.include_features = kwargs['include_features']
        self.augmented = kwargs['augmented']
        self.include_batch_size = kwargs['include_batch_size']
        self.load_csv()
        self.x, self.y = self.prepare()
        self.create_splits(**kwargs)
        self.num_ops = None

    def load_csv(self):
        raw_data = pd.read_csv(self.file_path)
        if self.subset == 'all':  # do nothing
            self.data = raw_data
        else:  # Options for subset are ['pretrained', 'simple'].
            self.data = raw_data.loc[raw_data['result.type'] == self.subset]


    def prepare(self, **kwargs):
        dense_features = ["input_size","output_size","units"]
        conv_features = ["input_size","output_size","filters", "kernel_size", "stride"]
        pool_features = ["input_size","output_size","filters (default=1)", "pool_size", "stride"]
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
            if self.augmented: #newer dataset with batch info
                self.x_batches = []
                for model_index, row in self.data.iterrows():
                    print(f"Processing Model {model_index + 1}/{len(self.data)}")
                    # if model_index > 50:
                    #    continue
                    try:
                        model = tf.keras.models.model_from_json(row['result.model'])
                        # model = tf.keras.models.model_from_json(ast.literal_eval(row['model'])['layers']['config']['layers'])
                        power = float(row['result.power'])
                        self.x_batches.append(float(row['config.batch_size']))
                        # if np.abs(power) > 1e-5:
                        # continue
                    except:
                        print(f"Error: Model {row['result.name']} with ID {row['_id']} could not be imported.")
                        continue
                    model_x = []
                    for layer in model.layers:

                        # layer_config = layer.get_config()
                        if layer.__class__.__name__ == 'Functional':
                            for sublayer in layer.layers:
                                model_x.append([sublayer.__class__.__name__,
                                                sublayer.input_shape,
                                                sublayer.output_shape])
                        else:
                            model_x.append([layer.__class__.__name__,
                                            layer.input_shape,
                                            layer.output_shape])
                    x.append(model_x)
                    y.append(power)

            else:#older dataset
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

        if self.include_batch_size:
            for idx, layer in enumerate(x):
                for feat in layer:
                    feat.append(self.x_batches[idx])

        x = tf.keras.preprocessing.sequence.pad_sequences(x, padding='post')

        return x, y

    def create_splits(self, *args, **kwargs):
        validation_split = kwargs['validation_split']
        test_split = kwargs['test_split']
        self.x_train, self.x_test, self.y_train, self.y_test = split(self.x, self.y,
                                                                     split_ratio=test_split,
                                                                     shuffle=True, seed=123)
        if validation_split != False:
            self.x_train, self.x_validation, self.y_train, self.y_validation = split(self.x_train, self.y_train,
                                                                                     split_ratio=validation_split,
                                                                                     shuffle=False, seed=None)
