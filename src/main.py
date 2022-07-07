"""
1. Create dataset class instance
2. Get train/validation/test splits
3. Create model class instance
4. Training
    - Overfitting Test
    - Full Training
5. Inference
6. Evaluation
"""
import os
import importlib
import random
import numpy as np
import matplotlib.pyplot as plt
import csv
import yaml
from pathlib import Path

yaml_path = "/home/denizkorkmaz/PycharmProjects/TUM/SS22/green-ml-daml/src/run_config_layerwise.yaml" # TODO: Needs to be set manually!
config = yaml.safe_load(Path(yaml_path).read_text())

def main():
    dataset_class_name = config['dataset_class']
    dataset_module = importlib.import_module(config[dataset_class_name]['module'])
    dataset_class = getattr(dataset_module, config[dataset_class_name]['class'])
    dataset = dataset_class(**config[dataset_class_name]['params'])

    x_train, y_train = dataset.get_train_set()
    x_val, y_val = dataset.get_validation_set()
    x_test, y_test = dataset.get_test_set()
    if x_val is None and y_val is None:
        print(f"Dataset:\nTraining:\tx: {x_train.shape}\ty: {y_train.shape}\nValidation:\tx: {None}\ty: {None}\nTest:\tx: {x_test.shape}\ty: {y_test.shape}\n")
    else:
        print(
            f"Dataset:\nTraining:\tx: {x_train.shape}\ty: {y_train.shape}\nValidation:\tx: {x_val.shape}\ty: {y_val.shape}\nTest:\tx: {x_test.shape}\ty: {y_test.shape}\n")

    # Create the model.
    model_class_name = config['model_class']
    model_module = importlib.import_module(config[model_class_name]['module'])
    model_class = getattr(model_module, config[model_class_name]['class'])
    model = model_class(**config[model_class_name]['params'])

    # Overfitting.
    random_index = random.randint(0, len(y_train)-1)
    x_overfit, y_overfit = np.expand_dims(x_train[random_index], axis=0), np.expand_dims(y_train[random_index], axis=0)
    model.train(x_train=x_overfit, y_train=y_overfit, x_val=None, y_val=None)
    y_predicted_overfit = model.predict(x_overfit)
    print(f"Overfitting:\tTarget: {y_overfit.item()}\tPrediction: {y_predicted_overfit.item()}")

    # Create the model again and train.
    model = model_class(**config[model_class_name]['params'])
    model.train(x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val)

    # Inference on the trained model
    y_predicted = model.predict(x_test=x_test).flatten()
    print("Y_predicted:\n",y_predicted)
    print("Y_test:\n",y_test)

    # Evaluation
    evaluation_module = importlib.import_module(config['evaluation']['module'])
    metrics = config['evaluation']['metrics']
    splits = {
        'train': (x_train, y_train),
        'val': (x_val, y_val),
        'test': (x_test, y_test)
    }
    loss_results = {}
    for split in splits:
        x_tmp, y_tmp = splits[split]
        if x_tmp is None or y_tmp is None:
            continue
        cur_loss_results = {}
        for metric in metrics:
            func = getattr(evaluation_module, metric)
            cur_loss_results[metric] = func(y_tmp, model.predict(x_test=x_tmp).flatten())
        loss_results[split] = cur_loss_results

    if hasattr(model, 'history'):
        history = model.history
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        plt.figure(figsize=(8, 8))
        plt.plot(loss, label='Training Loss')
        plt.plot(val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.ylabel(config[model_class_name]['params']['loss'])
        plt.ylim([0, 1])
        plt.title('Training and Validation Loss')
        plt.xlabel('epoch')
        plt.show()

    results = {
        'dataset':config[dataset_class_name]['class'],
        'subset':config[dataset_class_name]['params']['subset'],
        'target_layer':config[dataset_class_name]['params']['target_layer'] if 'target_layer' in config[dataset_class_name]['params'] else None,
        'training_dataset':len(y_train),
        'validation_dataset':len(y_val) if config[dataset_class_name]['params']['validation_split'] is not False else 0,
        'test_dataset':len(y_test),
        'model_name':config[model_class_name]['class'],
        'n_features': x_train.shape[1],
        'degree':config[model_class_name]['params']['degree'] if 'degree' in config[model_class_name]['params'] else None,
        'activation': config[model_class_name]['params']['activation'] if 'activation' in config[model_class_name]['params'] else None,
        'batch_size':config[model_class_name]['params']['batch_size'] if 'batch_size' in config[model_class_name]['params'] else None,
        'num_epochs_overfit':config[model_class_name]['params']['num_epochs_overfit'] if 'num_epochs_overfit' in config[model_class_name]['params'] else None,
        'overfitting_target':y_overfit.item(),
        'overfitting_prediction':y_predicted_overfit[0].item(),
        'num_epochs_training':config[model_class_name]['params']['num_epochs'] if 'num_epochs' in config[model_class_name]['params'] else None,
        'loss':config[model_class_name]['params']['loss'] if 'loss' in config[model_class_name]['params'] else None,
        'leaning_rate':config[model_class_name]['params']['lr'] if 'lr' in config[model_class_name]['params'] else None,
        'training_loss':model.history.history['loss'][-1] if hasattr(model, 'history') else None,
        'val_loss':model.history.history['val_loss'][-1] if hasattr(model, 'history') else None,
        'mae-train':loss_results['train']['mae'] if 'train' in loss_results else None,
        'mse-train':loss_results['train']['mse'] if 'train' in loss_results else None,
        'rmse-train':loss_results['train']['rmse'] if 'train' in loss_results else None,
        'rmspe-train':loss_results['train']['rmspe'] if 'train' in loss_results else None,
        'mae-val': loss_results['val']['mae'] if 'val' in loss_results else None,
        'mse-val': loss_results['val']['mse'] if 'val' in loss_results else None,
        'rmse-val': loss_results['val']['rmse'] if 'val' in loss_results else None,
        'rmspe-val': loss_results['val']['rmspe'] if 'val' in loss_results else None,
        'mae-test': loss_results['test']['mae'] if 'test' in loss_results else None,
        'mse-test': loss_results['test']['mse'] if 'test' in loss_results else None,
        'rmse-test': loss_results['test']['rmse'] if 'test' in loss_results else None,
        'rmspe-test': loss_results['test']['rmspe'] if 'test' in loss_results else None
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