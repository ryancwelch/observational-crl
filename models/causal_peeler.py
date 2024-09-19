import numpy as np
import torch
import gurobipy as gp
from gurobipy import GRB

def qcqp_solver(H_diff_sym, epsilon = 1e-2):

    epsilon = threshold(H_diff_sym) + epsilon
    # print('Threshold: ', epsilon)

    m = gp.Model("qcqp")
    m.setParam('OutputFlag', 0)
    
    x = m.addMVar(shape=H_diff_sym.shape[1], lb=-GRB.INFINITY, name="x")
    m.setObjective(0, GRB.MINIMIZE)
    
    for i in range(H_diff_sym.shape[0]):
        m.addConstr(x @ H_diff_sym[i] @ x <= epsilon, name=f"qc{i}_upper")
        m.addConstr(x @ H_diff_sym[i] @ x >= -epsilon, name=f"qc{i}_lower")
    
    m.addConstr(x @ x <= 1.01, name="norm_upper")
    m.addConstr(x @ x >= 0.99, name="norm_lower")
    
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

def threshold(H_diff_sym):
    m = gp.Model("abs_qcqp")
    m.setParam('OutputFlag', 0)
    
    x = m.addMVar(shape=H_diff_sym.shape[1], lb=-GRB.INFINITY, ub=GRB.INFINITY, name="x")
    t = m.addVar(name="t")

    m.setObjective(t, GRB.MINIMIZE)

    m.addConstr(x @ x <= 1.01, name="norm_upper")
    m.addConstr(x @ x >= 0.99, name="norm_lower")

    for i in range(H_diff_sym.shape[0]):
        xi = x @ (H_diff_sym[i] @ x)
        m.addConstr(t >= xi, name=f"quad_pos_{i}")
        m.addConstr(t >= -xi, name=f"quad_neg_{i}")

    m.optimize()

    if m.status == GRB.OPTIMAL:
        return t.X
    else:
        return None

def fill_matrix(A):
    n,m = A.shape
    if m == n: return m, A
    
    for _ in range(n - m):
        while True:
            new_col = np.random.rand(n, 1)
            new_col /= np.linalg.norm(new_col)
            
            test_matrix = np.hstack((A, new_col))
            if np.linalg.matrix_rank(test_matrix) == A.shape[1] + 1:
                A = test_matrix
                break
    return m, A


def G_inv_estimator_e(H_diff_sym, epsilon=1e-2):
    return fill_matrix(qcqp_solver(H_diff_sym, epsilon))

def symmetrize(H_diff):
    return np.array([(H_diff[i]+H_diff[i].T)/2.0 for i in range(H_diff.shape[0])])

def remove_layer(X, jacobian_estimator):
    if not isinstance(X, torch.Tensor):
        X = torch.Tensor(X)

    print('Calculating Jacobian Estimates...')
    if X.shape[1] == 1: #last layer trivially a linear combination
        return np.empty((X.shape[0],0)), X.detach().numpy()

    J_X = jacobian_estimator(X).detach().numpy()
    J_X_bar = np.mean(J_X, axis=0)
    J_X_diff = J_X - J_X_bar
    J_X_diff_sym = symmetrize(J_X_diff)
    print('Done.')
    print()
    norms = np.linalg.norm(J_X_diff_sym, ord='fro', axis=(1, 2))
    norm_threshold = np.percentile(norms, 75)
    if J_X_diff_sym.shape[0]*0.75 > 4000:
        filtered_indices = np.argsort(norms)[:4000]
    else: 
        filtered_indices = norms <= norm_threshold
    J_X_diff_sym = J_X_diff_sym[filtered_indices]
    
    print('Finding Optimal G_hat...')
    num_leafs, G_hat = G_inv_estimator_e(J_X_diff_sym)
    print('Done.')
    print()
    
    U_hat = (np.linalg.inv(G_hat)@(X.detach().numpy().T)).T
    leafs = U_hat[:,:num_leafs]
    new_X = U_hat[:,num_leafs:]
    
    return new_X, leafs

def identify(X, jacobian_estimator):
    U_dim = X.shape[1] #Can be a passed parameter, once you determine latent dimension
    layers = []
    U_estimates = np.empty((X.shape[0],0))
    while X.shape[1]:
        print('-----------------------------------------------------------')
        print(f'Removing Layer {len(layers)}.')
        print('-----------------------------------------------------------')

        X, leafs = remove_layer(X, jacobian_estimator)
        print(f'Peeled off {leafs.shape[1]} nodes.')
        layers.append([U_dim-U_estimates.shape[1]-i-1 for i in range(leafs.shape[1])])
        U_estimates = np.hstack((leafs, U_estimates))
        
    return U_estimates, layers[::-1]

