import numpy as np

def mse(act, pred):
    act, pred = np.array(act), np.array(pred)
    diff = pred - act
    differences_squared = diff ** 2
    mean_diff = differences_squared.mean()
    return mean_diff

def rmse(act, pred):
    act, pred = np.array(act), np.array(pred)
    diff = pred - act
    differences_squared = diff ** 2
    mean_diff = differences_squared.mean()
    rmse_val = np.sqrt(mean_diff)
    return rmse_val

def mae(act, pred):
    act, pred = np.array(act), np.array(pred)
    diff = pred - act
    abs_diff = np.absolute(diff)
    mean_diff = abs_diff.mean()
    return mean_diff

def rmspe(y_true, y_pred):
    '''
    Compute Root Mean Square Percentage Error between two arrays.
    '''
    eps = 1e-12
    loss = np.sqrt(np.mean(np.square(((y_true - y_pred) / (y_true+eps))), axis=0))

    return loss