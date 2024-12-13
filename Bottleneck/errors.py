import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from sklearn import preprocessing

# Mean absolute error
def mae(ground_truth, predicted):
    mae = (abs(ground_truth - predicted)).mean().mean()
    return mae

#Mean Square error
def mse(ground_truth, predicted):
    mse = ((ground_truth - predicted) ** 2).mean().mean()
    return mse

#Root Mean Squared Error
def rmse(ground_truth, predicted):
    rmse = ((ground_truth - predicted) ** 2).mean().mean() ** 0.5
    return rmse

#Mean Absolute Percentage Error
def mape(ground_truth, predicted):
    #epsilon = 1e-12
    mape= (abs(ground_truth - predicted) / (abs(ground_truth)))
    mape = np.where(np.isinf(mape), 0, mape)
    mape = mape.mean().mean() * 100
    return mape

#R-Squared
def r_squared(ground_truth,predicted):
    mean_ground_truth = ground_truth.mean().mean()
    r_squared = 1 - ((ground_truth - predicted) ** 2).mean().mean() / ((ground_truth - mean_ground_truth) ** 2).mean().mean()
    return r_squared

def compiled_errors(ground_truth, predicted):
    _mae = mae(ground_truth,predicted)
    _mse = mse(ground_truth,predicted)
    _rmse = rmse(ground_truth,predicted)
    _mape = mape(ground_truth,predicted)
    _r_squared = r_squared(ground_truth,predicted)
    return _mae, _mse, _rmse, _mape, _r_squared
