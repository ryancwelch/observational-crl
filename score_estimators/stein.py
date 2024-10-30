"""
Code taken/adapted from https://github.com/py-why/dodiscover
"""

import torch
import numpy as np


def heuristic_kernel_width(X):
    X_diff = X.unsqueeze(1)-X
    D = torch.norm(X_diff, dim=2, p=2)
    s = D.flatten().median()
    return s

def hessian_col(X_diff, G, col, eta, K, s):
    # Stein estimate
    Gv = torch.einsum("i,ij->ij", G[:, col], G)
    nabla2vK = torch.einsum("ik,ikj,ik->ij", X_diff[:, :, col], X_diff, K) / s**4
    nabla2vK[:, col] -= torch.einsum("ik->i", K) / s**2
    H_col = -Gv + torch.matmul(torch.inverse(K + eta * torch.eye(X_diff.shape[0])), nabla2vK)
    return H_col

def stein_hess(X, eta_G=0.01, eta_H=0.01):
    n,d = X.shape

    s = heuristic_kernel_width(X.detach())
    X_diff = X.unsqueeze(1)-X
    K = torch.exp(-torch.norm(X_diff, dim=2, p=2)**2 / (2 * s**2)) /s # /s ?
    nablaK = -torch.einsum("kij,ik->kj", X_diff, K) / s**2
    G = torch.matmul(torch.inverse(K + eta_G * torch.eye(n)), nablaK)

    H = torch.stack([hessian_col(X_diff, G, col, eta_H, K, s) for col in range(d)], dim=0)
    return H.transpose(0, 1)

def fast_stein_hess(X, eta_G=0.01, eta_H=0.01):
    device = X.device
    n,d = X.shape
    norm = torch.cdist(X, X, p=2)
    s = torch.median(norm.flatten())
    X_diff = X.unsqueeze(1)-X
    K = torch.exp(-norm**2 / (2 * s**2)) /s
    nablaK = -torch.einsum("kij,ik->kj", X_diff, K) / s**2
    G = torch.linalg.solve(K + eta_G * torch.eye(n, device=device), nablaK)
    Gv = torch.einsum("ij,ik->ijk", G, G)
    nabla2vK = torch.einsum("ikl,ikj,ik->ijl", X_diff, X_diff, K) / s**4
    diagonal_adjustment = torch.einsum("ij->i", K) / s**2
    for c in range(d):
        nabla2vK[:, c, c] -= diagonal_adjustment

    nabla2vK_flat = nabla2vK.reshape(n, -1)
    inv_term_flat = torch.linalg.solve(K + eta_H * torch.eye(n, device=device), nabla2vK_flat) 
    inv_term = inv_term_flat.reshape(n, d, d)
    inv_term = 0.5 * (inv_term + inv_term.permute(0, 2, 1))

    return -Gv + inv_term
