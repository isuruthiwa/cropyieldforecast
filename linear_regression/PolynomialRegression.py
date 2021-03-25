import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


def fit_model(X_tr, Y_tr, X_te, Y_te):
    transformer = PolynomialFeatures(degree=2, include_bias=False)
    x_ = transformer.fit_transform(np.reshape(X_tr, (1467, 392)))

    model = LinearRegression().fit(x_, Y_tr)

    x_pred = transformer.transform(np.reshape(X_te, (32, 392)))
    prediction = model.predict(x_pred)
    error = np.subtract(Y_te, prediction)
    print(model.intercept_)
    print(model.coef_)
    print(prediction)
    print(error)
