"""
1. Define dataset_path
2. Create dataset
3. Obtain splits
4. Get model
5. Train model
6. Predict
7. Evaluate

"""
import importlib

config = {
    'dataset_params':{
        'dataset_module': 'src.dataset.data.dataset',
        'dataset_classes':{
            'LayerWiseDataset': {
                'init_params': ['file_path', 'subset'],
                'prepare_params': ['target_layer'],
                'values': {
                    'file_path': '/home/denizkorkmaz/PycharmProjects/TUM/SS22/green-ml-daml/src/dataset/dataset_layerwise.csv',
                    'subset': 'all',
                    'target_layer': 'conv'
                }
            },
            'ModelWiseDataset': {
                'init_params': ['file_path', 'subset'],
                'prepare_params': [],
                'values': {
                    'file_path': '/home/denizkorkmaz/PycharmProjects/TUM/SS22/green-ml-daml/src/dataset/dataset_layerwise.csv',
                    'subset': 'all'
                }
            }
        },
        'validation_split':False,
        'test_split': 0.3
    },
    'model_params':{
        'model_module': 'src.models.polynomial_regression',
        'model_classes': {
            'polynomial_regression': {
                'init_params': ['degree'],
                'values':{
                    'degree':2
                }
            }
        },
    },
    'evaluation_params':{
        'evaluation_module': 'src.models.metrics',
        'losses': ['mae', 'mse', 'rmse', 'rmspe', 'r2'],
        'my_losses': ['mse','rmse','rmspe','r2']
    }
}

dataset_class = 'LayerWiseDataset'
dataset_params = config['dataset_params']
dataset_class_params = config['dataset_params']['dataset_classes'][dataset_class]

model_class = 'polynomial_regression'
model_params = config['model_params']
model_class_params = config['model_params']['model_classes'][model_class]

evaluation_params = config['evaluation_params']

# TODO: Collecting the training history and saving the training plots.
def main():
    # Import the dataset and convert the csv file into numpy arrays (not preprocessed yet).
    dataset_module = importlib.import_module(dataset_params['dataset_module'])
    dataset = dataset_module.LayerWiseDataset(*[dataset_class_params['values'][arg] for arg in dataset_class_params['init_params']])
    x,y = dataset.prepare(*[dataset_class_params['values'][arg] for arg in dataset_class_params['prepare_params']])

    # Preprocess the dataset and split into the subsets of training/validation/test.
    x,y = dataset.preprocessing(x,y)
    x_train, x_test, y_train, y_test  = dataset_module.split(x, y, split_ratio=dataset_params['test_split'], shuffle=True, seed=123)
    if dataset_params['validation_split'] != False:
        x_train, x_val, y_train, y_val = dataset_module.split(x_train, y_train, split_ratio=dataset_params['validation_split'], shuffle=True, seed=123)

    # Create the model.
    model_module = importlib.import_module(model_params['model_module'])
    model = model_module.PolynomialRegressor(*[model_class_params['values'][arg] for arg in model_class_params['init_params']])

    # Train the model.
    model.train(x_train=x_train, y_train=y_train, x_val=None, y_val=None)

    # Inference on the trained model
    y_predicted = model.predict(x_test=x_test)

    # Evaluation
    evaluation_module = importlib.import_module(evaluation_params['evaluation_module'])
    my_losses = evaluation_params['my_losses']
    loss_results = []
    for loss in my_losses:
        func = getattr(evaluation_module, loss)
        loss_results.append(func(y_test, y_predicted))
    print(my_losses)
    print(loss_results)


if __name__ == '__main__':
    main()