import numpy as np
import torch
import gurobipy as gp
from gurobipy import GRB 
from oracle import *

def qcqp_solver(H_diff_sym, epsilon = 5e-2):
    
    m = gp.Model("qcqp")
    # m.setParam('FeasibilityTol', 1e-3)
    
    x = m.addMVar(shape=H_diff_sym.shape[1], lb=-GRB.INFINITY, name="x")
    
    m.setObjective(0, GRB.MINIMIZE)
    
    for i in range(H_diff_sym.shape[0]):
        m.addConstr(x @ H_diff_sym[i] @ x <= epsilon, name=f"qc{i}_upper")
        m.addConstr(x @ H_diff_sym[i] @ x >= -epsilon, name=f"qc{i}_lower")
        
    # m.addConstr(x @ x <= 1.1, name="norm_upper")
    # m.addConstr(x @ x >= 0.9, name="norm_lower")
    m.addConstr(x @ x == 1, name="norm_constr")
    
    m.update()
    
    solutions = np.empty((H_diff_sym.shape[1],0))
    while True:
        m.Params.NonConvex = 2
        m.setParam('OutputFlag', 0)
        m.optimize()
        if m.status == GRB.OPTIMAL:
            print('Optimal solution found:')
            sol = x.X
            print(sol)
            print()
            solutions = np.hstack((solutions, sol.reshape(-1, 1)))
            m.addConstr(x @ sol == 0, name=f"orth_const_{solutions.shape[1]}")
            m.update()
        elif m.status == GRB.INFEASIBLE:
            print('No feasible solution found.')
            return solutions
        else:
            print('Optimization was stopped with status', m.status)
            return solutions
        

    
# def fill_full_rank_matrix(A):
#     """
#     Given a matrix A (n x m) with m <= n, add column vectors of norm 1 such that the matrix is n x n and full rank.
#     """
#     n,m = A.shape
#     if m == n: return A
    
#     columns_to_add = n - m
    
#     for _ in range(columns_to_add):
#         while True:
#             new_col = np.random.rand(n, 1)
#             new_col /= np.linalg.norm(new_col)
            
#             test_matrix = np.hstack((A, new_col))
#             if np.linalg.matrix_rank(test_matrix) == A.shape[1] + 1:
#                 A = test_matrix
#                 break

#     return A

def fill_full_rank_matrix(A):
    """
    Given a matrix A (n x m) with m <= n, add column vectors of norm 1 such that the matrix is n x n and full rank.
    """
    n,m = A.shape
    if m == n: return m, A
    
    columns_to_add = n - m
    
    for _ in range(columns_to_add):
        while True:
            new_col = np.random.rand(n, 1)
            new_col /= np.linalg.norm(new_col)
            
            test_matrix = np.hstack((A, new_col))
            if np.linalg.matrix_rank(test_matrix) == A.shape[1] + 1:
                A = test_matrix
                break

    return m, A

def G_inv_estimator(H_diff_sym, epsilon = 5e-4):
    return fill_full_rank_matrix(qcqp_solver(H_diff_sym, epsilon))

def symmetrize(H_diff):
    return np.array([(H_diff[i]+H_diff[i].T)/2.0 for i in range(H_diff.shape[0])])


def remove_layer(graph, U, G):
    #Estimate Jacobian of the score
    
    print('Calculating Jacobian Estimates...')
    X = ((G@(U.T)).T)
    if X.shape[1] == 1: #last layer trivially a linear combination
        graph.remove_sinks()
        return graph, np.empty((X.shape[0],0)), np.empty((0,0)), X
    
    J_X = H_X(graph, U, G).detach().numpy()
    J_X_bar = np.mean(J_X, axis=0)
    J_X_diff = J_X - J_X_bar
    J_X_diff_sym = symmetrize(J_X_diff)
    print('Done.')
    print()
    
    print('Finding Optimal G_hat...')
    num_leafs, G_hat = G_inv_estimator(J_X_diff_sym)
    print('Done.')
    print()
    
    print('U_hat as a linear combination of U:')
    beta_inv = np.linalg.inv(G_hat)@(G.detach().numpy())
    print(beta_inv)
    U_hat = (np.linalg.inv(G_hat)@(X.T)).T
    print()
    
    leafs = U_hat[:,0:num_leafs]
    leafs_indices = graph.get_sinks()
    non_leaf_indices = [i for i in range(graph.nnodes) if i not in leafs_indices]
    if num_leafs != len(leafs_indices):
        print('FOUND WRONG NUMBER OF LEAF NODES!')
    new_U = U[:,non_leaf_indices]
    new_G = torch.from_numpy(beta_inv[num_leafs:, non_leaf_indices].astype(np.float32))
    graph.remove_sinks()
    
    return graph, new_U, new_G, leafs

# def remove_layer(X, J_X):
#     if X.shape[1] == 1: #last layer trivially a linear combination
#         return np.empty((X.shape[0],0)), X,  None

#     J_X_bar = np.mean(J_X, axis=0)
#     J_X_diff = J_X - J_X_bar
#     J_X_diff_sym = symmetrize(J_X_diff)
    
#     print('Finding Optimal G_hat...')
#     num_leafs, G_hat = G_inv_estimator(J_X_diff_sym)
#     print('Done.')
#     print()
    
#     # print('Diagonal of Estimated Jacobian:')
#     J_U_hat = np.matmul(np.matmul(G_hat.T, J_X),G_hat)
    
#     # print('U_hat as a linear combination of U:')
#     # beta_inv = np.linalg.inv(G_hat)@(G.detach().numpy())
#     # print(beta_inv)
#     U_hat = (np.linalg.inv(G_hat)@(X.T)).T
#     new_X = U_hat[:,num_leafs:]
#     leafs = U_hat[:,:num_leafs]
#     new_J_X = J_U_hat[:,:,num_leafs:][:,num_leafs:,:]
    
#     return  new_X, leafs, new_J_X 


def identify(graph, U, G):
    layer = 1
    U_estimates = np.empty((U.shape[0],0))
    while graph.nnodes:
        print('-----------------------------------------------------------')
        print(f'Removing Layer {layer}.')
        print('-----------------------------------------------------------')

        graph, U, G, leafs = remove_layer(graph, U, G)
        print(f'Peeled off {leafs.shape[1]} nodes.')
        U_estimates = np.hstack((leafs, U_estimates))
        layer+=1
        
    return U_estimates

# def identify(graph, U, G):
#     X = ((G@(U.T)).T).detach().numpy()
#     J_X = H_X(graph, U, G).detach().numpy()

#     layer = 1
#     U_estimates = np.empty((U.shape[0],0))
#     while X.shape[1]:
#         print('-----------------------------------------------------------')
#         print(f'Removing Layer {layer}.')
#         print('-----------------------------------------------------------')

#         X, leafs, J_X = remove_layer(X, J_X)
#         print(f'Peeled off {leafs.shape[1]} nodes.')

#         U_estimates = np.hstack((leafs, U_estimates))
#         layer+=1
        
#     return U_estimates