import numpy as np
import torch
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

def estimate_noise(U_estimates, layers, model_instance):
    # model_instance is a callable function to create a instance of the desired nonlinear regression
    noise_estimates = np.zeros_like(U_estimates)
    noise_estimates[:, 0] = U_estimates[:, 0]
    upstream = [0]
    for layer in layers[1:]:
        for i in layer:
            X = U_estimates[:, upstream]
            y = U_estimates[:, i]
            model = model_instance()
            model.fit(X, y)
            predictions = model.predict(X)
            noise_estimates[:, i] = y - predictions
        upstream += layer
    return noise_estimates

#Example nonlinear regression instances
def MLPRegressorInstance():
    return MLPRegressor(hidden_layer_sizes=(100,), activation='relu', solver='adam', max_iter=500, random_state=42)

def SVRInstance():
    return SVR(kernel='rbf', C=1.0, epsilon=0.1)

def PolynomialInstance(degree=2):
    return Pipeline([
        ('poly_features', PolynomialFeatures(degree=degree)),
        ('linear_regression', LinearRegression())
    ])
