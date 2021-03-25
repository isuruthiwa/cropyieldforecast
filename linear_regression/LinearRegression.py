import numpy as np
import tensorflow as tf
from sklearn.linear_model import LinearRegression


def fit_model(X_tr, Y_tr, X_te, Y_te):
    rows, columns = X_tr.shape
    r_squared = np.empty(columns, dtype=float)
    predictions = np.empty(columns, dtype=object)

    for iteration in range(columns):
        model = LinearRegression().fit(np.reshape(X_tr[:, iteration], (-1, 1)), Y_tr)
        predictions[iteration] = model.predict(np.reshape(X_te[:, iteration], (-1, 1)))
        error = np.subtract(Y_te, predictions[iteration])
        print(error)


def predict_model(x):
    x = x[:, 3:]
