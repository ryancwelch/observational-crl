#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()
import torch
import time
import numpy as np
from generate_samples import SyntheticDAG
from scipy.spatial.distance import pdist, squareform
# from utils import *

def hessian_col(X_diff, G, col, eta, K, s):
    # Stein estimate
    Gv = torch.einsum("i,ij->ij", G[:, col], G)
    nabla2vK = torch.einsum("ik,ikj,ik->ij", X_diff[:, :, col], X_diff, K) / s**4
    nabla2vK[:, col] -= torch.einsum("ik->i", K) / s**2
    # H_col = -Gv + torch.matmul(torch.inverse(K + eta * torch.eye(X_diff.shape[0])), nabla2vK)
    H_col = -Gv + torch.linalg.solve(K + eta * torch.eye(X_diff.shape[0]), nabla2vK)
    return H_col


def Stein_hess_mat(X, eta_G=0.01, eta_H=0.01):
    n,d = X.shape
    s = heuristic_kernel_width(X.detach())
    X_diff = X.unsqueeze(1)-X
    # K = torch.exp(-torch.norm(X_diff, dim=2, p=2)**2 / (2 * s**2)) /s # /s ?
    K = torch.exp(-torch.cdist(X, X, p=2)**2 / (2 * s**2)) /s

    nablaK = -torch.einsum("kij,ik->kj", X_diff, K) / s**2
    # G = torch.matmul(torch.inverse(K + eta_G * torch.eye(n)), nablaK)
    G = torch.linalg.solve(K + eta_G * torch.eye(n), nablaK)
    # Compute the Hessian by column stacked together
    # H = np.stack([hessian_col(X_diff, G, col, eta, K, s).detach().numpy() for col in range(d)], axis=1)
    H = torch.stack([hessian_col(X_diff, G, col, eta_H, K, s) for col in range(d)], dim=0)

    return H.transpose(0, 1)

# def Stein_hess_all(X, eta_G=0.01, eta_H=0.01):
#     print('Starting')
#     n,d = X.shape
#     s = heuristic_kernel_width(X.detach())

#     print('1')
#     X_diff = X.unsqueeze(1)-X
#     K = torch.exp(-torch.cdist(X, X, p=2)**2 / (2 * s**2)) /s
#     nablaK = -torch.einsum("kij,ik->kj", X_diff, K) / s**2

#     print('2')
#     G = torch.linalg.solve(K + eta_G * torch.eye(n), nablaK)

#     print('3')
#     Gv = torch.einsum("ij,ik->ijk", G, G)
#     nabla2vK = torch.einsum("ikl,ikj,ik->ijl", X_diff, X_diff, K) / s**4
#     diagonal_adjustment = torch.einsum("ij->i", K) / s**2
#     for c in range(d):
#         nabla2vK[:, c, c] -= diagonal_adjustment

#     print('4')
#     nabla2vK_flat = nabla2vK.reshape(n, -1) # shape (n, d*d)
#     inv_term_flat = torch.linalg.solve(K + eta_H * torch.eye(n), nabla2vK_flat)  # shape (n, d*d)
#     inv_term = inv_term_flat.reshape(n, d, d)
#     inv_term = 0.5 * (inv_term + inv_term.permute(0, 2, 1))

#     return -Gv + inv_term

def Stein_hess(X, eta_G=0.01, eta_H=0.01):
    device = X.device #**
    n,d = X.shape
    # print(1)
    norm = torch.cdist(X, X, p=2)
    s = torch.median(norm.flatten())
    # s = heuristic_kernel_width(X.detach())
    # print(2)
    X_diff = X.unsqueeze(1)-X
    # print(3)
    K = torch.exp(-norm**2 / (2 * s**2)) /s
    # K = torch.exp(-torch.cdist(X, X, p=2)**2 / (2 * s**2)) /s
    # print(4)
    nablaK = -torch.einsum("kij,ik->kj", X_diff, K) / s**2
    # print(5)
    G = torch.linalg.solve(K + eta_G * torch.eye(n, device=device), nablaK)
    # print(6)
    Gv = torch.einsum("ij,ik->ijk", G, G)
    nabla2vK = torch.einsum("ikl,ikj,ik->ijl", X_diff, X_diff, K) / s**4
    diagonal_adjustment = torch.einsum("ij->i", K) / s**2
    for c in range(d):
        nabla2vK[:, c, c] -= diagonal_adjustment

    nabla2vK_flat = nabla2vK.reshape(n, -1) # shape (n, d*d)
    inv_term_flat = torch.linalg.solve(K + eta_H * torch.eye(n, device=device), nabla2vK_flat)  # shape (n, d*d)
    inv_term = inv_term_flat.reshape(n, d, d)
    inv_term = 0.5 * (inv_term + inv_term.permute(0, 2, 1))

    return -Gv + inv_term


def heuristic_kernel_width(X):
    # X_cpu = X.cpu().numpy()
    # D = pdist(X_cpu, 'euclidean')
    # s = np.median(D)
    # return torch.tensor(s, dtype=torch.float32)
    return torch.median(torch.cdist(X, X, p=2).flatten())

#################################

def sample_full_rank_matrix(n=3, lower=-10, upper=10):
    while True:
        matrix = torch.FloatTensor(n, n).uniform_(lower, upper)
        matrix.requires_grad_(True)
        matrix.retain_grad()
        if torch.det(matrix).item() != 0:
            return matrix

# layers = [[0], [1], [2], [3]]
# DAG_type = 'line'
# num_latent = 4
# num_samples = 100000
# lower_G = -1 #min possibel value in G or G_hat
# upper_G = 1 #max possible value in G or G_hat
# G = sample_full_rank_matrix(num_latent, lower_G, upper_G)+torch.eye(num_latent)

# graph = SyntheticDAG(num_latent, DAG_type)
# U,X,noises = graph.sample_scaled(G, num_samples)
# X = X.detach()

# start = time.time()
# Stein_hess_all(X)
# end = time.time()
# print('TIME TO RUN ON 100K SAMPLES: ', end-start)


# def heuristic_kernel_width_gpu(X, batch_size=50000):
#     device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
#     X = X.to(device)

#     n = X.size(0)
#     distances = []

#     # Process in smaller batches to avoid out-of-memory errors
#     for i in range(0, n, batch_size):
#         end_i = min(i + batch_size, n)
#         for j in range(0, n, batch_size):
#             end_j = min(j + batch_size, n)
#             with torch.no_grad():
#                 D_batch = torch.cdist(X[i:end_i], X[j:end_j], p=2)
#                 distances.append(D_batch.cpu().flatten())  # Move to CPU and flatten
#                 del D_batch  # Ensure memory is freed immediately
#                 torch.cuda.empty_cache()  # Clear the cache to free memory

#     # Concatenate all distances on the CPU
#     distances = torch.cat(distances)
#     s = distances.median().item()  # Compute the median
#     return s
