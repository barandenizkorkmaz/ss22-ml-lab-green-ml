from src.dataset.data import dataset
from src.dataset.data.dataset import LayerWiseDataset
from src.models.polynomial_regression import PolynomialRegression
from src.models.metrics import mae, mse, rmse, rmspe, r2

import numpy as np

models = dict()

ds = LayerWiseDataset(file_path='/home/denizkorkmaz/PycharmProjects/TUM/SS22/green-ml-daml/src/dataset/datasetGPU.csv', subset='pretrained')
x, y = ds.prepare(is_model_wise=False, target_layer="dense")
y = np.array(y, dtype=float)
x, y = ds.preprocessing(x, y)

x_train, x_test, y_train, y_test = dataset.split(x, y, split_ratio=0.2, shuffle=True, seed=123)

model_dense = PolynomialRegression(degree=2)
model_dense.train(x_train, y_train, x_val=None, y_val=None)
y_predicted = model_dense.predict(x_test=x_test).flatten()
rmspe_dense = rmspe(y_test, y_predicted).item()

models["dense"] = model_dense

x, y = ds.prepare(is_model_wise=False, target_layer="conv")

x_train, x_test, y_train, y_test = dataset.split(x, y, split_ratio=0.2, shuffle=True, seed=123)

model_conv = PolynomialRegression(degree=3)
model_conv.train(x_train, y_train, x_val=None, y_val=None)
y_predicted = model_conv.predict(x_test=x_test).flatten()
rmspe_conv = rmspe(y_test, y_predicted).item()

models["conv"] = model_conv

x, y = ds.prepare(is_model_wise=False, target_layer="pool")

x_train, x_test, y_train, y_test = dataset.split(x, y, split_ratio=0.2, shuffle=True, seed=123)

model_pool = PolynomialRegression(degree=1)
model_pool.train(x_train, y_train, x_val=None, y_val=None)
y_predicted = model_pool.predict(x_test=x_test).flatten()
rmspe_pool = rmspe(y_test, y_predicted).item()

models["pool"] = model_pool

del x_train, y_train

dataset = LayerWiseDataset(file_path='/home/denizkorkmaz/PycharmProjects/TUM/SS22/green-ml-daml/src/dataset/datasetGPU.csv', subset='pretrained')
x, y = dataset.prepare(is_model_wise=True, target_layer=None)
y_predicted = []

for model_data in x:
    consumption = []
    for layer_data in model_data:
        layer_type = layer_data[0]
        layer_features = np.array(layer_data[1:])
        layer_features = np.expand_dims(layer_features, axis=0)
        consumption.append(models[layer_type].predict(layer_features).item())
    y_predicted.append(sum(consumption)*1e-9)

y = np.array(y, dtype=float).flatten()
y_predicted = np.array(y_predicted, dtype=float).flatten()

print("y_actual:\n",y)
print("y_predicted:\n",y_predicted)

print(f"RMSPE of Trained Models\nDense: {rmspe_dense}\tConv: {rmspe_conv}\tPool: {rmspe_pool}")

# Evaluation
loss_fcs = {
    'mae':mae,
    'mse':mse,
    'rmse':rmse,
    'rmspe':rmspe,
    'r2':r2
}
loss_results = []
for loss_fn in loss_fcs:
    loss_results.append(loss_fcs[loss_fn](y, y_predicted))
loss_results = {loss_metric: loss_value for loss_metric, loss_value in zip(loss_fcs, loss_results)}
print(loss_results)
