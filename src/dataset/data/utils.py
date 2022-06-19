
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



