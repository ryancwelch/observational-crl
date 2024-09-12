from generate_samples import SyntheticDAG
from causal_peeler_e import *
from stein import *
import csv
import numpy as np
import torch
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

layers = [[0], [1], [2], [3]]

def noise_estimation(id_layers, layers):
    noise_estimates = np.zeros_like(id_layers)
    noise_estimates[:,0] = id_layers[:,0]
    upstream=[0]
    for layer in layers[1:]:
        for i in layer:
            degree = 2  # Degree of the polynomial features
            poly_features = PolynomialFeatures(degree=degree)
            X_poly = poly_features.fit_transform(id_layers[:,upstream])
            model = LinearRegression()
            model.fit(X_poly, id_layers[:,i])
            noise_estimates[:,i] = id_layers[:,i] - model.predict(X_poly)
        upstream+=layer
    return noise_estimates

def sample_full_rank_matrix(n=3, lower=-10, upper=10):
    while True:
        matrix = torch.FloatTensor(n, n).uniform_(lower, upper)
        matrix.requires_grad_(True)
        matrix.retain_grad()
        if torch.det(matrix).item() != 0:
            return matrix
        
def mac(noises, noises_estimates):
    correlation_coeffs = []
    for col in range(noises.shape[1]):
        correlation_matrix = np.corrcoef(noises[:, col], noises_estimates[:, col])
        # print(correlation_matrix)
        correlation_coeff = correlation_matrix[0, 1]
        correlation_coeffs.append(abs(correlation_coeff))
    return np.mean(correlation_coeffs)

def append_value_to_csv(value, num_samples):
    with open(f'tests/stein/{num_samples}.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([value])

DAG_type = 'line'
num_latent = 4
lower_G = -1 #min possibel value in G or G_hat
upper_G = 1 #max possible value in G or G_hat


num_samples_list = [100, 1000, 10000]
num_trials = 10

for num_samples in num_samples_list:
    for i in range(num_trials):
        print(f'Starting: {num_samples}, {i}')
        graph = SyntheticDAG(num_latent, DAG_type)
        G = sample_full_rank_matrix(num_latent, lower_G, upper_G)+torch.eye(num_latent)
        
        U, X, noises = graph.sample_scaled(G, num_samples)
        try:
            id_layers = identify_e(X, estimator='stein')
            noise_estimates = noise_estimation(id_layers, layers)
            append_value_to_csv(mac(noises, noise_estimates), num_samples)
        except:
            print('ERROR!!!')
            continue