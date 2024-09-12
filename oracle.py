import numpy as np
import torch


def gaussian_pdf(x, mu, var):
    return (1/torch.sqrt(2*np.pi*var))*torch.exp(-0.5*((x-mu)**2/var))

def p_u(graph, U):
    prod = 1.0
    for i in graph.top_order:
        prod *= gaussian_pdf(U[i], graph.coeffs[i]*graph.f(U[list(graph.DAG.parents_of(i))])+graph.means[i], graph.variances[i])
        #  graph.f(U[list(graph.DAG.parents_of(i))]))
    return prod

def log_pu(graph, U):
    return torch.log(p_u(graph, U))

def score_u(graph, U):
    return torch.autograd.grad(outputs=log_pu(graph, U), inputs=U, create_graph=True)[0]

def jacobian_score_u(graph, U):
    grad_log_pu = score_u(graph, U)
    hessian = torch.zeros(U.size(0), U.size(0))
    for i in range(U.size(0)):
        grad_grad_log_pu = torch.autograd.grad(outputs=grad_log_pu[i], inputs=U, create_graph=True)
        hessian[i] = grad_grad_log_pu[0]
    return hessian

# def H_X(graph, U, G):
#     # U = graph.descale(U)
#     jacobians = torch.zeros(U.shape[0], U.shape[1], U.shape[1])
#     for i in range(len(U)):
#         jacobians[i] = torch.inverse(G).T@jacobian_score_u(graph, U[i])@torch.inverse(G)
#     jacobians.requires_grad_(True)
#     jacobians.retain_grad()
#     return jacobians

def H_X(graph, U, G):
    J_U = H_U(graph, U)
    return torch.matmul(torch.matmul(torch.inverse(G).T, J_U), torch.inverse(G))

def H_U(graph, U):
    U.requires_grad_(True)
    U.retain_grad()
    jacobians = torch.zeros(U.shape[0], U.shape[1], U.shape[1])
    for i in range(len(U)):
        jacobians[i] = jacobian_score_u(graph, U[i])
    return jacobians
