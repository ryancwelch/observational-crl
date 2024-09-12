import numpy as np
import torch

from generate_samples import SyntheticDAG
from oracle import *
from causal_peeler_e import *
from causal_peeler import *
from stein import *
from ssm import *
import pandas as pd


def sample_full_rank_matrix(n=3, lower=-10, upper=10):
    while True:
        matrix = torch.FloatTensor(n, n).uniform_(lower, upper)
        matrix.requires_grad_(True)
        matrix.retain_grad()
        if torch.det(matrix).item() != 0:
            return matrix
        

DAG_type = 'random'
variance_max = 1
lower_G = -1 #min possibel value in G or G_hat
upper_G = 1 #max possible value in G or G_hat
score_est_steps = 10

#SSM

num_training_examples = [25000, 50000, 75000, 100000]
num_testing = 2000
num_latent_vals = [4,5,6]


for num_latent in num_latent_vals:
    G = sample_full_rank_matrix(num_latent, lower_G, upper_G)+torch.eye(num_latent)
    graph = SyntheticDAG(num_latent, DAG_type, variance_max)
    for num_training in num_training_examples:
        U = graph.get_samples(num_testing)
        U_tr = graph.get_samples(num_training)
        X = (G@U.T).T
        X_tr = (G@U_tr.T).T

        #True matrices
        J_X = H_X(graph, U, G)

        #Stein estimated matrices

        J_X_ssm = ssm_hess(X, X_tr)

        diff = J_X - J_X_ssm

        mse = (diff ** 2).mean(dim=(1, 2))

        frobenius_norm = torch.sqrt((diff ** 2).sum(dim=[1, 2]))

        data = {
            "MSE": mse.detach().numpy(),  # Convert tensor to numpy array
            "Frobenius Norm": frobenius_norm.detach().numpy()  # Convert tensor to numpy array
        }
        df = pd.DataFrame(data)
        df.to_csv(f"tests/ssm/nl_{num_latent}_ns_{num_training/100}K.csv", index=False)


#STEIN

# num_samples_vals = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000]
# num_latent_vals = [4,5,6]

# for num_latent in num_latent_vals:
#     G = sample_full_rank_matrix(num_latent, lower_G, upper_G)+torch.eye(num_latent)
#     graph = SyntheticDAG(num_latent, DAG_type, variance_max)
#     for num_samples in num_samples_vals:
#         U = graph.get_samples(num_samples)
#         X = (G@U.T).T

#         #True matrices
#         J_X = H_X(graph, U, G)

#         #Stein estimated matrices
#         J_X_st = Stein_hess_mat(X, eta=0.01)

#         diff = J_X - J_X_st

#         mse = (diff ** 2).mean(dim=(1, 2))

#         frobenius_norm = torch.sqrt((diff ** 2).sum(dim=[1, 2]))

#         data = {
#             "MSE": mse.detach().numpy(),  # Convert tensor to numpy array
#             "Frobenius Norm": frobenius_norm.detach().numpy()  # Convert tensor to numpy array
#         }
#         df = pd.DataFrame(data)
#         df.to_csv(f"tests/stein/nl_{num_latent}_ns_{num_samples}.csv", index=False)
            






