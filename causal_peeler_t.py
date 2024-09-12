import numpy as np
import torch
import gurobipy as gp
from gurobipy import GRB 
# from oracle import *
from oracle_laplace import *


def SER(true_matrices, estimated_matrices):
    total_true_norm = sum(np.linalg.norm(A, 'fro')**2 for A in true_matrices)
    total_error_norm = sum(np.linalg.norm(A - A_hat, 'fro')**2 for A, A_hat in zip(true_matrices, estimated_matrices))
    global_SER = total_true_norm / total_error_norm
    global_SER_dB = 10 * np.log10(global_SER)
    return global_SER, global_SER_dB

def perturb(true_matrices, desired_SER_dB):
    # Convert SER dB to linear scale
    desired_SER_linear = 10 ** (desired_SER_dB / 10)

    # Calculate total signal power
    total_true_norm = sum(np.linalg.norm(A, 'fro')**2 for A in true_matrices)

    # Calculate total noise power based on desired SER
    total_noise_power = total_true_norm / desired_SER_linear

    # Number of matrices
    n_matrices = len(true_matrices)

    # Calculate individual noise power if equally distributed
    individual_noise_power = total_noise_power / n_matrices
    # Generate noise matrices and estimated matrices
    noise_matrices = []
    estimated_matrices = []

    for A in true_matrices:
        # Generate random noise matrix
        noise = np.random.randn(*A.shape)
        # Scale the noise to have the correct Frobenius norm
        noise_norm = np.linalg.norm(noise, 'fro')
        scaled_noise = noise * (np.sqrt(individual_noise_power) / noise_norm)
        noise_matrices.append(scaled_noise)
        # Add noise to the true matrix to create the estimated matrix
        estimated_matrices.append(A + scaled_noise)

    return estimated_matrices

def qcqp_solver(H_diff_sym):
    
    epsilon = qcqp_solver_min(H_diff_sym)+1e-2

    m = gp.Model("qcqp")
    # m.setParam('FeasibilityTol', 1e-4)
    m.setParam('OutputFlag', 0)
    m.setParam('TimeLimit', 120)  
    
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
        # m.setParam('OutputFlag', 0)
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

def qcqp_solver_min(H_diff_sym):
    
    n,d,_ = H_diff_sym.shape

    m = gp.Model("abs_qcqp")
    m.setParam('OutputFlag', 0)
    m.setParam('FeasibilityTol', 1e-4)
    m.setParam('TimeLimit', 120)  
    
    x = m.addMVar(shape=d, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="x")

    t = m.addVar(name="t")

    m.setObjective(t, GRB.MINIMIZE)

    m.addConstr(x @ x == 1, name="norm_upper")
    # m.addConstr(x @ x >= 0.99, name="norm_lower")
    
    for i in range(n):
        xi_positive = gp.QuadExpr()
        xi_negative = gp.QuadExpr()
        for j in range(d):
            for k in range(d):
                xi_positive += x[j] * H_diff_sym[i, j, k] * x[k]
                xi_negative += -x[j] * H_diff_sym[i, j, k] * x[k]
        m.addConstr(t >= xi_positive, f"quad_pos_{i}")
        m.addConstr(t >= xi_negative, f"quad_neg_{i}")

    # Optimize the model
    m.optimize()

    # Extract the solution if optimal
    if m.status == GRB.OPTIMAL:
        # x_opt = [x[i].X for i in range(d)]
        # print(x.X)
        return t.X
    return None
    

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
    
def G_inv_estimator(H_diff_sym):
    return fill_full_rank_matrix(qcqp_solver(H_diff_sym))

def symmetrize(H_diff):
    return np.array([(H_diff[i]+H_diff[i].T)/2.0 for i in range(H_diff.shape[0])])


def remove_layer(graph, U, X, G, db):
    #Estimate Jacobian of the score
    if X.shape[1] == 1: #last layer trivially a linear combination
        graph.remove_sinks()
        return graph, np.empty((X.shape[0],0)), np.empty((X.shape[0],0)), np.empty((0,0)), X
    
    print('Calculating Jacobian Estimates...')
    # J_X_true = H_X(graph, U, G).detach().numpy()
    J_X = H_X(graph, U, G).detach().numpy()
    # J_X = perturb(J_X_true, db)
    # print(f'With SER dB: {SER(J_X_true, J_X)[1]}')
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
    new_X = U_hat[:,num_leafs:]
    leafs_indices = graph.get_sinks()
    non_leaf_indices = [i for i in range(graph.nnodes) if i not in leafs_indices]
    if num_leafs != len(leafs_indices):
        print('FOUND WRONG NUMBER OF LEAF NODES!')
    new_U = U[:,non_leaf_indices]
    new_G = torch.from_numpy(beta_inv[num_leafs:, non_leaf_indices].astype(np.float32))
    graph.remove_sinks()
    
    return graph, new_U, new_X, new_G, leafs


def identify(graph, U, X, G, db):
    layer = 1
    X = X.detach().numpy()
    U_estimates = np.empty((U.shape[0],0))
    while graph.nnodes:
        print('-----------------------------------------------------------')
        print(f'Removing Layer {layer}.')
        print('-----------------------------------------------------------')

        graph, U, X, G, leafs = remove_layer(graph, U, X, G, db)
        print(f'Peeled off {leafs.shape[1]} nodes.')
        U_estimates = np.hstack((leafs, U_estimates))
        layer+=1
        
    return U_estimates

