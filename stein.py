#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()
import torch
import numpy as np
from scipy.spatial.distance import pdist, squareform
# from utils import *

    
def Stein_hess(X, eta_G, eta_H, s = None):
    """
    Estimates the diagonal of the Hessian of log p_X at the provided samples points
    X, using first and second-order Stein identities
    """
    n, d = X.shape
    
    X_diff = X.unsqueeze(1)-X
    if s is None:
        D = torch.norm(X_diff, dim=2, p=2)
        s = D.flatten().median()
    K = torch.exp(-torch.norm(X_diff, dim=2, p=2)**2 / (2 * s**2)) / s
    
    nablaK = -torch.einsum('kij,ik->kj', X_diff, K) / s**2
    G = torch.matmul(torch.inverse(K + eta_G * torch.eye(n)), nablaK)
    
    nabla2K = torch.einsum('kij,ik->kj', -1/s**2 + X_diff**2/s**4, K)
    return -G**2 + torch.matmul(torch.inverse(K + eta_H * torch.eye(n)), nabla2K)

def compute_top_order(X, eta_G, eta_H, normalize_var=True, dispersion="var"):
    n, d = X.shape
    order = []
    active_nodes = list(range(d))
    for i in range(d-1):
        H = Stein_hess(X, eta_G, eta_H)
        if normalize_var:
            H = H / H.mean(axis=0)
        if dispersion == "var": # The one mentioned in the paper
            l = int(H.var(axis=0).argmin())
        elif dispersion == "median":
            med = H.median(axis = 0)[0]
            l = int((H - med).abs().mean(axis=0).argmin())
        else:
            raise Exception("Unknown dispersion criterion")
        order.append(active_nodes[l])
        active_nodes.pop(l)
        X = torch.hstack([X[:,0:l], X[:,l+1:]])
    order.append(active_nodes[0])
    order.reverse()
    return order
    

def Stein_hess_parents(X, s, eta, l): # If one wants to estimate the leaves parents based on the off-diagonal part of the Hessian
    n, d = X.shape
    
    X_diff = X.unsqueeze(1)-X
    K = torch.exp(-torch.norm(X_diff, dim=2, p=2)**2 / (2 * s**2)) / s
    
    nablaK = -torch.einsum('ikj,ik->ij', X_diff, K) / s**2
    G = torch.matmul(torch.inverse(K + eta * torch.eye(n)), nablaK)
    Gl = torch.einsum('i,ij->ij', G[:,l], G)
    
    nabla2lK = torch.einsum('ik,ikj,ik->ij', X_diff[:,:,l], X_diff, K) / s**4
    nabla2lK[:,l] -= torch.einsum("ik->i", K) / s**2
    
    return -Gl + torch.matmul(torch.inverse(K + eta * torch.eye(n)), nabla2lK)


def hessian_col(X_diff, G, col, eta, K, s):
    # Stein estimate
    Gv = torch.einsum("i,ij->ij", G[:, col], G)
    nabla2vK = torch.einsum("ik,ikj,ik->ij", X_diff[:, :, col], X_diff, K) / s**4
    nabla2vK[:, col] -= torch.einsum("ik->i", K) / s**2
    H_col = -Gv + torch.matmul(torch.inverse(K + eta * torch.eye(X_diff.shape[0])), nabla2vK)
    return H_col


def Stein_hess_mat(X, eta_G=0.01, eta_H=0.01):
    n,d = X.shape

    s = heuristic_kernel_width(X.detach())
    X_diff = X.unsqueeze(1)-X
    K = torch.exp(-torch.norm(X_diff, dim=2, p=2)**2 / (2 * s**2)) /s # /s ?
    nablaK = -torch.einsum("kij,ik->kj", X_diff, K) / s**2
    G = torch.matmul(torch.inverse(K + eta_G * torch.eye(n)), nablaK)

    # Compute the Hessian by column stacked together
    # H = np.stack([hessian_col(X_diff, G, col, eta, K, s).detach().numpy() for col in range(d)], axis=1)
    H = torch.stack([hessian_col(X_diff, G, col, eta_H, K, s) for col in range(d)], dim=0)
    return H.transpose(0, 1)

# def hessian_col(X_diff, G, col, eta, K, s):
#     # Stein estimate
#     Gv = torch.einsum("i,ij->ij", G[:, col], G)
#     nabla2vK = torch.einsum("ik,ikj,ik->ij", X_diff[:, :, col], X_diff, K) / s**4
#     nabla2vK[:, col] -= torch.einsum("ik->i", K) / s**2
#     H_col = -Gv + torch.matmul(torch.inverse(K + eta * torch.eye(X_diff.shape[0], dtype=X_diff.dtype, device=X_diff.device)), nabla2vK)
#     return H_col

# def Stein_hess_mat(X, eta_G=0.01, eta_H=0.01, alpha=1, c=1, degree=2):
#     n, d = X.shape
#     s = heuristic_kernel_width(X.detach())
#     X_diff = X.unsqueeze(1) - X
#     dot_products = torch.einsum("ik,jk->ij", X, X)
#     K = (alpha * dot_products + c)**degree
#     grad_K = d * alpha * (alpha * dot_products + c)**(degree - 1)
#     nablaK = torch.einsum('ij,jk->ik', grad_K, X)
#     # nablaK = degree * alpha * (alpha * dot_products + c)**(degree - 1) * X_diff
#     G = torch.matmul(torch.inverse(K + eta_G * torch.eye(n, dtype=X.dtype, device=X.device)), nablaK)

#     # Compute the Hessian by column stacked together
#     H = torch.stack([hessian_col(X_diff, G, col, eta_H, K, s) for col in range(d)], dim=0)
#     return H.transpose(0, 1)

# def hessian_col(X_diff, G, col, eta, K, s, degree=2, c=1, scaling_factor=1000, value_cap=100):
#     Gv = torch.einsum("i,ij->ij", G[:, col], G)
    
#     nabla2vK = torch.zeros((X_diff.shape[0], X_diff.shape[2]))
#     for j in range(X_diff.shape[2]):
#         # Implementing scaling and capping
#         scaled_X_diff_col = X_diff[:, :, col] / scaling_factor
#         scaled_X_diff_j = X_diff[:, :, j] / scaling_factor
#         term1 = (degree - 1) * degree * torch.einsum('ik,ik,ik->i', scaled_X_diff_col, scaled_X_diff_j, K) / (s**4)
#         if j == col:
#             term2 = degree * K[:, j] / scaling_factor
#         else:
#             term2 = torch.zeros(X_diff.shape[0])

#         nabla2vK[:, j] = torch.clamp(term1 - term2, min=-value_cap, max=value_cap)

#     regularized_K = K + eta * torch.eye(K.size(0))
#     H_col = -Gv + torch.matmul(torch.inverse(regularized_K), nabla2vK)
    
#     return H_col


# def Stein_hess_mat(X, eta_G=0.01, eta_H=0.01, c=1, degree=2, scaling_factor=1000, value_cap=113):
#     n,d = X.shape
#     s = heuristic_kernel_width(X.detach())
#     X_norm = (X - X.mean(dim=0)) / X.std(dim=0)
#     X_diff = X_norm.unsqueeze(1) - X_norm
#     # X_diff = X.unsqueeze(1)-X
#     # K = torch.exp(-torch.norm(X_diff, dim=2, p=2)**2 / (2 * s**2)) / s # /s ?
#     K = (torch.mm(X_norm, X_norm.t())+1.0)**2
#     # nablaK = -torch.einsum("kij,ik->kj", X_diff, K) / s**2
#     # nablaK = 
#     nablaK = torch.zeros((n, d))
#     K_ = (torch.mm(X_norm, X_norm.t()) + c)**(degree - 1)
#     for j in range(d):
#         nablaK[:, j] = torch.mv(K_, X_norm[:, j]) * degree
#     G = torch.matmul(torch.inverse(K + eta_G * torch.eye(n)), nablaK)
#     # print(G)
#     H = torch.stack([hessian_col(X_diff, G, col, eta_H, K, s, degree, c, scaling_factor, value_cap) for col in range(d)], dim=0)
#     return H.transpose(0, 1)

def heuristic_kernel_width(X):
    #ORIGINAL ############
        # X_diff = X.unsqueeze(1)-X
        # D = torch.norm(X_diff, dim=2, p=2)
        # s = D.flatten().median()
        # return s
    ###############

    X_cpu = X.cpu().numpy()
    D = pdist(X_cpu, 'euclidean')
    s = np.median(D)
    return torch.tensor(s, dtype=torch.float32)

# def heuristic_kernel_width(X):
#     X_diff = X.unsqueeze(1)-X
#     D = torch.norm(X_diff, dim=2, p=2)
#     s = D.flatten().median()
#     return s

def fullAdj2Order(A):
    order = list(A.sum(axis=1).argsort())
    order.reverse()
    return order


def num_errors(order, adj):
    err = 0
    for i in range(len(order)):
        err += adj[order[i+1:], order[i]].sum()
    return err


def Stein_grad(X, eta, s=None): # Not used
    n, d = X.shape

    X_diff = X.unsqueeze(1)-X
    if s is None:
        D = torch.norm(X_diff, dim=2, p=2)
        s = D.flatten().median()
    K = torch.exp(-torch.norm(X_diff, dim=2, p=2)**2 / (2 * s**2)) / s

    nablaK = -torch.einsum('kij,ik->kj', X_diff, K) / s**2
    return torch.matmul(torch.inverse(K + eta * torch.eye(n)), nablaK)