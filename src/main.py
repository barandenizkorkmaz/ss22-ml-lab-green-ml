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
import importlib
import random
import numpy as np
import matplotlib.pyplot as plt

config = {
    'dataset_params':{
        'dataset_module': 'src.dataset.data.dataset',
        'dataset_classes':{
            'LayerWiseDataset': {
                'init_params': {
                    'file_path': '/home/denizkorkmaz/PycharmProjects/TUM/SS22/green-ml-daml/src/dataset/datasetGPU.csv', # Needs to be set manually
                    'subset': 'simple'
                },
                'prepare_params': {
                    'target_layer': 'dense'
                }
            }
        },
        'validation_split':0.2,
        'test_split': 0.2
    },
    'model_params':{
        'model_module': 'src.models.mlp_layerwise',
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
                    'loss': 'mse', # init_params must always include loss!
                    'lr': 0.001,
                    'n_features': 3 # Needs to be set inside the main. Please don't push your model-specific changes into the repository!
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

dataset_class = 'LayerWiseDataset' # Needs to be set manually
dataset_params = config['dataset_params']
dataset_class_params = config['dataset_params']['dataset_classes'][dataset_class]

model_class = 'mlp_layerwise' # Needs to be set manually
model_params = config['model_params']
model_class_params = config['model_params']['model_classes'][model_class]

evaluation_params = config['evaluation_params']

# TODO: Collecting the training history and saving the training plots.
def main():
    # Import the dataset and convert the csv file into numpy arrays (not preprocessed yet).
    dataset_module = importlib.import_module(dataset_params['dataset_module'])
    print("Arg:",dataset_class_params['init_params'])
    dataset = dataset_module.LayerWiseDataset(**dataset_class_params['init_params']) # Needs to be changed manually
    x,y = dataset.prepare(**dataset_class_params['prepare_params'])
    print(f"x.shape: {x.shape}\ty.shape: {y.shape}")

    # Preprocess the dataset and split into the subsets of training/validation/test.
    x,y = dataset.preprocessing(x,y)
    x_train, x_test, y_train, y_test  = dataset_module.split(x, y, split_ratio=dataset_params['test_split'], shuffle=True, seed=123)
    print(f"Dataset:\tSize of Entire Dataset: {len(y)}\tSize of Training Data: {len(y_train)}\tSize of Test Data: {len(y_test)}")
    if dataset_params['validation_split'] != False:
        x_train, x_val, y_train, y_val = dataset_module.split(x_train, y_train, split_ratio=dataset_params['validation_split'], shuffle=True, seed=123)

    # Create the model.
    model_module = importlib.import_module(model_params['model_module'])
    model = model_module.MLPLW(**model_class_params['init_params']) # Needs to be changed manually

    # Overfitting.
    random_index = random.randint(0, len(y_train))
    x_overfit, y_overfit = np.expand_dims(x_train[random_index], axis=0), np.expand_dims(y_train[random_index], axis=0)
    model.train(x_train=x_overfit, y_train=y_overfit, x_val=None, y_val=None)
    y_predicted = model.predict(x_overfit)
    print(f"Overfitting:\tTarget: {y_overfit.item()}\tPrediction: {y_predicted.item()}")

    # Create the model again and train.
    model_module = importlib.import_module(model_params['model_module'])
    model = model_module.MLPLW(**model_class_params['init_params']) # Needs to be changed manually
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
    print(my_losses)
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


if __name__ == '__main__':
    main()