from src.dataset.data.dataset import LayerWiseDataset, split
import pandas as pd

file_path = '/home/denizkorkmaz/PycharmProjects/TUM/SS22/green-ml-daml/src/dataset/datasetGPU.csv'
subsets = ['simple','pretrained','all']
target_layers = ['dense','conv','pool']

features = {
    'dense': ["input_shape", "output_shape", "hidden_size"],
    'conv': ["input_shape", "output_shape", "filters", "kernel_size", "stride"],
    'pool': ["input_shape", "output_shape", "filters (default=1)", "pool_size", "stride"]
}

for subset in subsets:
    for target_layer in target_layers:
        print(f"Current Iteration:\tSubset: {subset}\tTarget Layer: {target_layer}")
        dataset = LayerWiseDataset(file_path=file_path, subset=subset)
        x,y = dataset.prepare(is_model_wise = False, target_layer = target_layer)
        x, y = dataset.preprocessing(x, y)

        x_train, x_test, y_train, y_test = split(x, y, split_ratio=0.2, shuffle=True, seed=123)

        splits = {
            'train': (x_train, y_train),
            'test': (x_test, y_test)
        }

        for _split in splits:
            x_split, y_split = splits[_split]
            results = {}
            for i, feature in enumerate(features[target_layer]):
                results[feature] = x_split[:,i]
            results['energy_consumption'] = y_split
            df = pd.DataFrame(data=results, columns=results.keys())
            df.to_csv(f"layerwise-{subset}-{target_layer}-{_split}", encoding='utf-8', index=False)
