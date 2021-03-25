import numpy as np
from sklearn.linear_model import LinearRegression


def fit_model(X_tr, Y_tr, X_te, Y_te):
    model = LinearRegression().fit(np.reshape(X_tr, (1467, 392)), Y_tr)
    prediction = model.predict(np.reshape(X_te, (32, 392)))
    error = np.subtract(Y_te, prediction)
    print(model.intercept_)
    print(model.coef_)
    print(prediction)
    print(error)
