from generate_samples import SyntheticDAG
from causal_peeler_t import *
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

def append_value_to_csv(value, i,j):
    with open('tests/SER_test.csv', 'r', newline='') as file:
        reader = list(csv.reader(file))
    print(f'Writing to {reader[i][0]} at location {j}')
    reader[i][j] = value
    with open('tests/SER_test.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(reader)

DAG_type = 'line'
num_latent = 4
lower_G = -1 #min possibel value in G or G_hat
upper_G = 1 #max possible value in G or G_hat

# for i in range(10):
#     ser = ((i+1)*5)+100
#     for j in range(10):
#         print(f"SER DB: {ser}, 'TEST #: {j}")
#         graph = SyntheticDAG(num_latent, DAG_type)
#         G = sample_full_rank_matrix(num_latent, lower_G, upper_G)+torch.eye(num_latent)
#         U,X,noises = graph.sample_scaled(G, 3000)
#         try:
#             id_layers = identify(graph, U, X, G, ser)
#             noise_estimates = noise_estimation(id_layers, layers)
#             MAC = mac(noises, noise_estimates)
#             append_value_to_csv(MAC, 26+i, j+1)
#         except:
#             print('ERROR!')
#             continue

