"""
1. Define dataset_path
2. Create dataset (includes prepare)
3. Preprocess the dataset
4. Obtain splits
5. Get model
6. Train model
7. Predict
8. Evaluate
"""
import os
import importlib
import random
import numpy as np
import matplotlib.pyplot as plt
import csv

config = {
    'dataset_params':{
        'dataset_module': 'src.dataset.data.dataset',
        'dataset_classes':{
            'LayerWiseDataset': {
                'init_params': {
                    'file_path': '/home/denizkorkmaz/PycharmProjects/TUM/SS22/green-ml-daml/src/dataset/datasetGPU.csv', #TODO: Needs to be set manually
                    'subset': 'simple'
                },
                'prepare_params': {
                    'target_layer': 'dense'
                }
            }
        },
        'validation_split':False,
        'test_split': 0.2
    },
    'model_params':{
        'model_module': 'src.models.polynomial_regression',
        'model_classes': {
            'polynomial_regression': {
                'init_params': {
                    'degree':2
                }
            },
            'mlp_layerwise':{
                'init_params': {
                    'batch_size': 16,
                    'num_epochs': 100,
                    'loss': 'mse', # init_params must include loss!
                    'lr': 0.001,
                    'n_features': 4, # Can be set inside the main or given as hard-coded.
                    'num_epochs_overfit': 1000
                },
            }
        },
    },
    'evaluation_params':{
        'evaluation_module': 'src.models.metrics',
        'losses': ['mae', 'mse', 'rmse', 'rmspe', 'r2'],
        'my_losses': ['mae','mse','rmse','rmspe','r2']
    }
}

dataset_class = 'LayerWiseDataset' #TODO: Needs to be set manually
dataset_params = config['dataset_params']
dataset_class_params = config['dataset_params']['dataset_classes'][dataset_class]

model_class = 'polynomial_regression' #TODO: Needs to be set manually
model_params = config['model_params']
model_class_params = config['model_params']['model_classes'][model_class]

evaluation_params = config['evaluation_params']

# TODO: Collecting the training history and saving the training plots.
def main():
    # Import the dataset and convert the csv file into numpy arrays (not preprocessed yet).
    dataset_module = importlib.import_module(dataset_params['dataset_module'])
    dataset = dataset_module.LayerWiseDataset(**dataset_class_params['init_params']) #TODO: Needs to be changed manually
    x,y = dataset.prepare(**dataset_class_params['prepare_params'])

    # Preprocess the dataset and split into the subsets of training/validation/test.
    x,y = dataset.preprocessing(x,y)
    x_train, x_test, y_train, y_test  = dataset_module.split(x, y, split_ratio=dataset_params['test_split'], shuffle=True, seed=123)
    print(f"Dataset:\tSize of Entire Dataset: {len(y)}\tSize of Training Data: {len(y_train)}\tSize of Test Data: {len(y_test)}")
    if dataset_params['validation_split'] != False:
        x_train, x_val, y_train, y_val = dataset_module.split(x_train, y_train, split_ratio=dataset_params['validation_split'], shuffle=True, seed=123)
    else:
        x_val, y_val = None, None

    # Create the model.
    model_module = importlib.import_module(model_params['model_module'])
    model = model_module.PolynomialRegression(**model_class_params['init_params']) #TODO: Needs to be changed manually

    # Overfitting.
    random_index = random.randint(0, len(y_train))
    x_overfit, y_overfit = np.expand_dims(x_train[random_index], axis=0), np.expand_dims(y_train[random_index], axis=0)
    model.train(x_train=x_overfit, y_train=y_overfit, x_val=None, y_val=None)
    y_predicted_overfit = model.predict(x_overfit)
    print(f"Overfitting:\tTarget: {y_overfit.item()}\tPrediction: {y_predicted_overfit.item()}")

    # Create the model again and train.
    model_module = importlib.import_module(model_params['model_module'])
    model = model_module.PolynomialRegression(**model_class_params['init_params']) #TODO: Needs to be changed manually
    model.train(x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val)

    # Inference on the trained model
    y_predicted = model.predict(x_test=x_test).flatten()

    # Evaluation
    evaluation_module = importlib.import_module(evaluation_params['evaluation_module'])
    my_losses = evaluation_params['my_losses']
    loss_results = []
    for loss in my_losses:
        func = getattr(evaluation_module, loss)
        loss_results.append(func(y_test, y_predicted))
    loss_results = {loss_metric:loss_value for loss_metric,loss_value in zip(my_losses,loss_results)}
    print(loss_results)

    if hasattr(model, 'history'):
        history = model.history
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        plt.figure(figsize=(8, 8))
        plt.plot(loss, label='Training Loss')
        plt.plot(val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.ylabel(model_class_params['init_params']['loss'])
        plt.ylim([0, max(max(loss),max(val_loss))])
        plt.title('Training and Validation Loss')
        plt.xlabel('epoch')
        plt.show()

    results = {
        'dataset':dataset_class,
        'subset':dataset_class_params['init_params']['subset'],
        'target_layer':dataset_class_params['prepare_params']['target_layer'] if 'target_layer' in dataset_class_params['prepare_params'] else None,
        'training_dataset':len(y_train),
        'validation_dataset':len(y_val) if dataset_params['validation_split'] is not False else 0,
        'test_dataset':len(y_test),
        'model_name':model_class,
        'model':model.to_json() if callable(getattr(model, 'to_json', None)) else None,
        'degree':model_class_params['init_params']['degree'] if 'degree' in model_class_params['init_params'] else None,
        'batch_size':model_class_params['init_params']['batch_size'] if 'batch_size' in model_class_params['init_params'] else None,
        'num_epochs_overfit':model_class_params['init_params']['num_epochs_overfit'] if 'num_epochs_overfit' in model_class_params['init_params'] else None,
        'overfitting_target':y_overfit.item(),
        'overfitting_prediction':y_predicted_overfit[0].item(),
        'num_epochs_training':model_class_params['init_params']['num_epochs'] if 'num_epochs' in model_class_params['init_params'] else None,
        'loss':model_class_params['init_params']['loss'] if 'loss' in model_class_params['init_params'] else None,
        'leaning_rate':model_class_params['init_params']['lr'] if 'lr' in model_class_params['init_params'] else None,
        'training_loss':model.history.history['loss'][-1] if hasattr(model, 'history') else None,
        'val_loss':model.history.history['val_loss'][-1] if hasattr(model, 'history') else None,
        'mae':loss_results['mae'],
        'mse':loss_results['mse'],
        'rmse':loss_results['rmse'],
        'rmspe':loss_results['rmspe']
    }
    isFileExists = os.path.isfile('daml-green-ml-results.csv')
    with open('daml-green-ml-results.csv', 'a') as csvfile:
        fieldnames = list(results.keys())
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if isFileExists == False:
            writer.writeheader()
        writer.writerow(results)


if __name__ == '__main__':
    main()