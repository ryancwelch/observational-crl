import numpy as np
import torch
import gurobipy as gp
from gurobipy import GRB 
# from stein import *
from fast_stein import *
from ssm import *

def qcqp_solver(H_diff_sym, epsilon = 5e-4):
    
    m = gp.Model("qcqp")
    # m.setParam('FeasibilityTol', 1e-3)
    
    x = m.addMVar(shape=H_diff_sym.shape[1], lb=-GRB.INFINITY, name="x")
    
    m.setObjective(0, GRB.MINIMIZE)
    
    for i in range(H_diff_sym.shape[0]):
        m.addConstr(x @ H_diff_sym[i] @ x <= epsilon, name=f"qc{i}_upper")
        m.addConstr(x @ H_diff_sym[i] @ x >= -epsilon, name=f"qc{i}_lower")
        
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
        
def qcqp_solver_e(J_diff_sym):

    frobenius_norms = np.linalg.norm(J_diff_sym, ord='fro', axis=(1, 2))
    threshold = np.percentile(frobenius_norms, 75)
    if J_diff_sym.shape[0]*0.75 > 4000:
        filtered_indices = np.argsort(frobenius_norms)[:4000]
    else: 
        filtered_indices = frobenius_norms <= threshold
    J_diff_sym = J_diff_sym[filtered_indices]

    m = gp.Model("qcqp")
    # m.setParam('FeasibilityTol', 1e-3)

    x = m.addMVar(shape=J_diff_sym.shape[1], lb=-GRB.INFINITY, name="x")
    s = m.addMVar(shape=J_diff_sym.shape[0], lb=0, name="s")

    m.setObjective(gp.quicksum(s[i]**2 for  i in range(J_diff_sym.shape[0])), GRB.MINIMIZE)
    
    for i in range(J_diff_sym.shape[0]):
        # m.addConstr(x @ J_diff_sym[i] @ x == s[i], name=f"qc{i}")
        m.addConstr(x @ J_diff_sym[i] @ x <= s[i], name=f"qc{i}_upper")
        m.addConstr(x @ J_diff_sym[i] @ x >= -s[i], name=f"qc{i}_lower")

    m.addConstr(x @ x == 1, name="norm_constr")

    # Additional constraints to force y_i = 1 if s_i < threshold
    # m.addConstr(gp.quicksum(y) >= J_diff_sym.shape[0]*0.75)

    m.update()
    m.Params.NonConvex = 2
    m.setParam('OutputFlag', 0)
    m.setParam('TimeLimit', 300)  # Set a reasonable time limit (300 seconds)
    m.setParam('MIPGap', 0.1)  # Adjust gap tolerance if appropriate

    m.optimize()

    # Results
    solutions = np.empty((J_diff_sym.shape[1],0))
    if m.status == GRB.OPTIMAL:
        print("Optimal solution found.")
        sol = x.X
        print(f"x: {sol}")
        print()
        # print(f"s: {s.X}")
        return np.hstack((solutions, sol.reshape(-1, 1)))
    else:
        print("Optimal solution not found.")
        return solutions

    # epsilon = qcqp_solver_min(H_diff_sym)+1e-4
    # print('Threshold: ', epsilon)

    # epsilon = epsilon + 0.5
    # m = gp.Model("qcqp")
    # m.setParam('OutputFlag', 0)
    # # m.setParam('FeasibilityTol', 1e-3)
    
    # x = m.addMVar(shape=H_diff_sym.shape[1], lb=-GRB.INFINITY, name="x")
    
    # m.setObjective(0, GRB.MINIMIZE)
    
    # for i in range(H_diff_sym.shape[0]):
    #     m.addConstr(x @ H_diff_sym[i] @ x <= epsilon, name=f"qc{i}_upper")
    #     m.addConstr(x @ H_diff_sym[i] @ x >= -epsilon, name=f"qc{i}_lower")
        
    # m.addConstr(x @ x <= 1.01, name="norm_upper")
    # m.addConstr(x @ x >= 0.99, name="norm_lower")
    
    # m.update()
    
    # solutions = np.empty((H_diff_sym.shape[1],0))
    # while True:
    #     m.Params.NonConvex = 2
    #     m.setParam('OutputFlag', 0)
    #     m.optimize()
    #     if m.status == GRB.OPTIMAL:
    #         print('Optimal solution found:')
    #         sol = x.X
    #         print(sol)
    #         print()
    #         solutions = np.hstack((solutions, sol.reshape(-1, 1)))
    #         m.addConstr(x @ sol == 0, name=f"orth_const_{solutions.shape[1]}")
    #         m.update()
    #     elif m.status == GRB.INFEASIBLE:
    #         print('No feasible solution found.')
    #         return solutions
    #     else:
    #         print('Optimization was stopped with status', m.status)
    #         return solutions
        


def qcqp_solver_min(H_diff_sym):
    
    n,d,_ = H_diff_sym.shape

    m = gp.Model("abs_qcqp")
    m.setParam('OutputFlag', 0)
    # m.setParam('FeasibilityTol', 1e-3)
    
    x = m.addMVar(shape=d, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="x")

    t = m.addVar(name="t")

    m.setObjective(t, GRB.MINIMIZE)

    m.addConstr(x @ x <= 1.01, name="norm_upper")
    m.addConstr(x @ x >= 0.99, name="norm_lower")
    
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
    else:
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

def G_inv_estimator_e(H_diff_sym, epsilon = 5e-4):
    return fill_full_rank_matrix(qcqp_solver_e(H_diff_sym))

def symmetrize(H_diff):
    return np.array([(H_diff[i]+H_diff[i].T)/2.0 for i in range(H_diff.shape[0])])


def remove_layer_e(X, X_tr=None, estimator='stein'):
    eta_G=0.10911070444147038
    eta_H=0.9033964848783047
    # c=5.826745534656855
    # scaling_factor = 6562.681373045313
    # degree=2
    #Estimate Jacobian of the score
    if not isinstance(X, torch.Tensor):
        X = torch.Tensor(X)
    if X_tr is not None:
        X_tr = torch.Tensor(X_tr)
    print('Calculating Jacobian Estimates...')

    if X.shape[1] == 1: #last layer trivially a linear combination
        if X_tr is not None:
            return np.empty((X.shape[0],0)), np.empty((X_tr.shape[0],0)), X.detach().numpy()
        return np.empty((X.shape[0],0)), None, X.detach().numpy()

    if estimator=='stein':
        J_X = Stein_hess(X, eta_G=eta_G, eta_H=eta_H).detach().numpy() #, c=c, degree=2, scaling_factor=scaling_factor, value_cap=125).detach().numpy()
    else:
        J_X = ssm_hess(X, X_tr).detach().numpy()
    print(J_X.shape)
    J_X_bar = np.mean(J_X, axis=0)
    J_X_diff = J_X - J_X_bar
    J_X_diff_sym = symmetrize(J_X_diff)
    print('Done.')
    print()
    
    print('Finding Optimal G_hat...')
    num_leafs, G_hat = G_inv_estimator_e(J_X_diff_sym)
    print('Done.')
    print()
    
    # print('Diagonal of Estimated Jacobian:')
    # J_U_hat = np.matmul(np.matmul(G_hat.T, J_X),G_hat)
    # J_U_hat_var = np.var(J_U_hat, axis=0)
    # print(np.diag(J_U_hat_var))
    # num_zero_diag = (np.diag(J_U_hat_var) < 1e-4).sum()
    # print()
    X = X.detach().numpy()
    if X_tr is not None:
        X_tr = X_tr.detach().numpy()

    U_hat = (np.linalg.inv(G_hat)@(X.T)).T
    # leafs_indices = [index for index, value in enumerate(np.var(J_U_hat, axis=0)) if value < 1e-4]
    leafs = U_hat[:,:num_leafs]
    # non_leaf_indices = [i for i in range(X.shape[1]) if i not in leafs_indices]
    new_X = U_hat[:,num_leafs:]
    if X_tr is not None:
        new_X_tr = (np.linalg.inv(G_hat)@(X_tr.T)).T[:,num_leafs:]
    else:
        new_X_tr=None
    
    return new_X, new_X_tr, leafs


def identify_e(X, X_tr=None, estimator='stein'):
    print('Starting...')
    layer = 1
    U_estimates = np.empty((X.shape[0],0))
    while X.shape[1]:
        print('-----------------------------------------------------------')
        print(f'Removing Layer {layer}.')
        print('-----------------------------------------------------------')

        X, X_tr, leafs = remove_layer_e(X, X_tr, estimator)
        print(f'Peeled off {leafs.shape[1]} nodes.')
        U_estimates = np.hstack((leafs, U_estimates))
        layer+=1
        
    return U_estimates
    
