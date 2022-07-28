import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

from src.dataset.experiments.flopEvaluation import get_flops

def map_to_ints(x):
    """
    Args:
        x: Dataset consisting of operation type as one feature

    Returns: Dictionary mapping op to int
    """
    ops = [x[j][i][0] for j in range(len(x)) for i in range(len(x[j]))]
    op_set = set(ops)
    op_to_int = dict((o, i) for i, o in enumerate(op_set))
    return op_to_int


def get_op_index(x, map):
    return map[x]


def indexify(dataset):
    """
    Args:
        dataset:

    Returns: Converted dataset with indices in place of operation types
    """
    mapping = map_to_ints(dataset)
    for model in dataset:
        for layer in model:
            layer[0] = [get_op_index(layer[0], mapping)]
    return len(mapping), dataset


def one_hot_encoder(to_encode: list, map: dict) -> list:
    onehot_encoded = [0] * len(map)
    onehot_encoded[map[to_encode]] += 1
    return onehot_encoded


def one_hotify(dataset):
    """
    Args:
        dataset:

    Returns: Converted dataset with one-hot-encodings in place of operation types
    """
    mapping = map_to_ints(dataset)
    for model in dataset:
        for layer in model:
            layer[0] = one_hot_encoder(layer[0], mapping)
    return dataset


def shape_converter(to_convert) -> list:
    """
    Args:
        to_convert: Raw input/output shapes as [None,a,b,c] or [(None,a,b,c)] or two shapes at concat layers

    Returns: List as [Dim1, Dim2, Dim3], uses 1 in missing dims

    """
    if type(to_convert) == list:
        to_convert = to_convert[0]
    size = len(to_convert)
    converted = list(to_convert[1:size])
    while(len(converted) < 3):
        converted.append(1)
    return converted


def convert_shapes(dataset):
    """
    Applies shape_converter to all input and output pairs
    """
    for model in dataset:
        for layer in model:
            layer[1], layer[2] = shape_converter(layer[1]), shape_converter(layer[2])
    return dataset


def flatten(dataset):
    """
    Args:
        dataset: Dataset with layer features as [[one_hot_encoding],[input_shape],[output_shape]]

    Returns: Dataset with layer features as [one_hot_encoding, input_shape, output_shape]
    """
    flattened = []
    for model in dataset:
        mod_int = []
        for layer in model:
            mod_int.append([layer[i][j] for i in range(len(layer)) for j in range(len(layer[i]))])
        flattened.append(mod_int)
    return flattened

def extract_layer_features(layer):
    """
    Extract the features for a given type of layer.

    Returns:
        if successful:
            list (features), string (layer type)
        else:
            False

    """
    current_layer_type = layer.__class__.__name__.lower()
    layer_config = layer.get_config()
    if "dense" in current_layer_type:
        input_size = np.prod([*[dim for dim in layer.input_shape if dim != None]])
        output_size = np.prod([*[dim for dim in layer.output_shape if dim != None]])
        hidden_size = layer_config["units"]
        #num_flops = get_flops(model_from_layer(layer))
        return [input_size, output_size, hidden_size], "dense"
    elif "conv" in current_layer_type:
        input_size = np.prod([*[dim for dim in layer.input_shape if dim != None]])
        output_size = np.prod([*[dim for dim in layer.output_shape if dim != None]])
        try:
            num_filters = layer_config["filters"]
        except:  # Possibly depth-wise conv
            num_filters = layer.output_shape[-1]
        kernel_size = np.prod([*[dim for dim in layer_config["kernel_size"] if dim != None]])
        stride = layer_config["strides"][0]
        #num_flops = get_flops(model_from_layer(layer))
        return [input_size, output_size, num_filters, kernel_size, stride], "conv"
    elif "pool" in current_layer_type:
        try:
            input_size = np.prod([*[dim for dim in layer.input_shape if dim != None]])
            output_size = np.prod([*[dim for dim in layer.output_shape if dim != None]])
            num_filters = 1
            pool_size = np.prod([*[dim for dim in layer_config["pool_size"] if dim != None]])
            stride = layer_config["strides"][0]
            #num_flops = get_flops(model_from_layer(layer))
            return [input_size, output_size, num_filters, pool_size, stride], "pool"
        except:  # Ignore
            return False
    elif current_layer_type == "inputlayer":
        try:
            input_size = np.prod([*[dim for dim in layer_config['batch_input_shape'] if dim != None]])
            return [input_size], "inputlayer"
        except:
            return False
    elif "pad" in current_layer_type:
        try:
            input_size = np.prod([*[dim for dim in layer.input_shape if dim != None]])
            output_size = np.prod([*[dim for dim in layer.output_shape if dim != None]])
            padding = layer_config["padding"]
            padding = sum(flatten_iterable(padding))
            return [input_size, output_size, padding], "pad"
        except:  # Ignore
            return False
    elif "normalization" in current_layer_type:
        try:
            input_size = np.prod([*[dim for dim in layer.input_shape if dim != None]])
            output_size = np.prod([*[dim for dim in layer.output_shape if dim != None]])
            return [input_size, output_size], "normalization"
        except:  # Ignore
            return False
    elif "activation" in current_layer_type or "relu" in current_layer_type:
        try:
            input_size = np.prod([*[dim for dim in layer.input_shape if dim != None]])
            return [input_size], "activation"
        except:  # Ignore
            return False
    elif "rescaling" in current_layer_type:
        try:
            input_size = np.prod([*[dim for dim in layer.input_shape if dim != None]])
            return [input_size], "rescaling"
        except:  # Ignore
            return False
    elif "reshape" in current_layer_type:
        try:
            input_size = np.prod([*[dim for dim in layer.input_shape if dim != None]])
            target_shape = np.prod([*[dim for dim in flatten_iterable(layer_config['target_shape']) if dim != None]])
            return [input_size, target_shape], "reshape"
        except:  # Ignore
            return False
    elif current_layer_type == "dropout":
        try:
            input_size = np.prod([*[dim for dim in layer.input_shape if dim != None]])
            rate = layer_config["rate"]
            return [input_size, rate], "dropout"
        except:  # Ignore
            return False
    elif current_layer_type == "add":
        try:
            output_size = np.prod([*[dim for dim in layer.output_shape if dim != None]])
            return [output_size], "add"
        except:  # Ignore
            return False
    elif current_layer_type == "multiply":
        try:
            output_size = np.prod([*[dim for dim in layer.output_shape if dim != None]])
            return [output_size], "multiply"
        except:  # Ignore
            return False
    elif current_layer_type == "concatenate":
        try:
            output_size = np.prod([*[dim for dim in layer.output_shape if dim != None]])
            return [output_size], "concatenate"
        except:  # Ignore
            return False
    return False

def pad_to_dense(M):
    """Appends the minimal required amount of zeroes at the end of each
     array in the jagged array `M`, such that `M` looses its jagedness."""

    maxlen = max(len(r) for r in M)

    Z = np.zeros((len(M), maxlen))
    for enu, row in enumerate(M):
        Z[enu, :len(row)] += row
    return Z

def flatten_iterable(l):
    """
    Flattens any number of iterables recursively.
    """
    import collections
    for el in l:
        if isinstance(el, collections.Iterable) and not isinstance(el, str):
            for sub in flatten_iterable(el):
                yield sub
        else:
            yield el

def model_from_layer(layer):
    """
    Builds a model from a single layer.
    """
    input_shape = layer.input_shape
    model = tf.keras.Sequential()
    model.add(layer)
    model.build(input_shape=input_shape)
    return model