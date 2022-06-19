import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def mse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred, squared=True)

def rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred, squared=False)

def mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)

def rmspe(y_true, y_pred):
    eps = 1e-12
    return np.sqrt(np.mean(np.square(((y_true - y_pred) / (y_true+eps))), axis=0))

def r2(y_true, y_pred):
    return r2_score(y_true, y_pred)