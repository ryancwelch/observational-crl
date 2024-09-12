from argparse import Namespace
import numpy as np
import matplotlib.pyplot as plt
import torch
import sys
sys.path.append('../actlearn_optint/optint')
from data import synthetic_instance, gen_dag
# from actlearn_optint.optint.run import run
from visualize import draw
from random import random

# from graphical_models.classes.dags.dag import DAG


# def gen_dag(nnodes, DAG_type):

# 	DAG_gen = {
# 		'random': random_graph,
# 		'barabasialbert': barabasialbert_graph,
# 		'line': line_graph,
# 		'path': path_graph,
# 		'instar': instar_graph,
# 		'outstar': outstar_graph,
# 		'tree':  tree_graph,
# 		'complete': complete_graph,
# 		'chordal': chordal_graph,
# 		'rootedtree': rooted_tree_graph,
# 		'cliquetree': cliquetree_graph
# 	}.get(DAG_type, None)
# 	assert DAG_type is not None, 'Unsuppoted DAG type!'

# 	return DAG_gen(nnodes)	

class SyntheticDAG:
    def __init__(self, nnodes, DAG_type, f=lambda x: torch.norm(x, p=2)**2, seed=1234):
        # f=lambda x: np.linalg.norm(x, ord=2)**2
        np.random.seed(seed)
        self.nnodes = nnodes
        self.f = f
        self.coeffs = [1 for _ in range(nnodes)] #coefficient of f(parents(u))
        self.means = [0 for _ in range(nnodes)] #mean of gaussian
        self.variances = [1 for _ in range(nnodes)] #variance of gaussian
        self.DAG = gen_dag(nnodes, DAG_type)
        self.top_order = self.DAG.topological_sort()
        self.removed_nodes = []


    def apply_f(self,x):
        results = []
        for row in x:
            results.append(self.f(row))
        return torch.tensor(results)

    def sample_scaled(self, G, num_samples=500):
        U = torch.zeros(num_samples, self.nnodes)
        noises = torch.zeros(num_samples, self.nnodes)
        for i in self.top_order:
            scale = random()
            noise = torch.randn(num_samples)*scale
            x = self.apply_f(U[:, list(self.DAG.parents_of(i))]) + noise

            a = (1/(torch.max(x) - torch.min(x)))
            b = -torch.min(x)/(torch.max(x) - torch.min(x))
            U[:,i] = a*x + b
            noises[:,i] = a*noise
            self.coeffs[i] = a
            self.means[i] = b
            self.variances[i] = (a**2)*(scale**2) #temporary - get nan values for 0.0 in U
            noise = a*noise
        mask = torch.any(U == 0.0, dim=1)
        U = U[~mask]
        noises = noises[~mask]
        return U, (G@U.T).T, noises

    # def apply_f(self,x):
    #     results = []
    #     for row in x:
    #         results.append(self.f(torch.tensor(row)))
    #     return np.array(results)

    # def sample_scaled(self, G, num_samples=500):
    #     U = np.zeros((num_samples, self.nnodes), dtype=np.float64)
    #     noises = np.zeros((num_samples, self.nnodes), dtype=np.float64)
    #     for i in self.top_order:
    #         scale = random()
    #         noise = np.random.randn(num_samples)*scale #np.random.randn(num_samples)*scale
    #         x = self.apply_f(U[:, list(self.DAG.parents_of(i))]) + noise
    #         a = (1/(np.max(x) - np.min(x)))
    #         b = -np.min(x)/(np.max(x) - np.min(x))
    #         U[:,i] = a*x + b
    #         noises[:,i] = a*noise
    #         self.coeffs[i] = a
    #         self.means[i] = b
    #         self.variances[i] = (a**2)*(scale**2) #temporary - get nan values for 0.0 in U
    #     mask = np.any(U == 0.0, axis=1)
    #     U = U[~mask]
    #     noises = noises[~mask]
    #     return U, (G@U.T).T, noises
        # U = torch.zeros(num_samples, self.nnodes)
        # noises = torch.zeros(num_samples, self.nnodes)
        # for i in self.top_order:
        #     scale = random()
        #     noise = torch.randn(num_samples)*scale
        #     x = self.apply_f(U[:, list(self.DAG.parents_of(i))]) + noise
        #     print(x)
        #     a = (1/(torch.max(x) - torch.min(x)))
        #     b = -torch.min(x)/(torch.max(x) - torch.min(x))
        #     U[:,i] = a*x + b
        #     noises[:,i] = a*noise
        #     self.coeffs[i] = a
        #     self.means[i] = b
        #     self.variances[i] = (a**2)*(scale**2) #temporary - get nan values for 0.0 in U
        # # mask = torch.any(U == 0.0, dim=1)
        # # U = U[~mask]
        # # noises = noises[~mask]
        # return U, (G@U.T).T, noises

    # def sample(self):
    #     # noise = [torch.normal(0, torch.sqrt(variance)).requires_grad_() for variance in self.variances]
    #     noise = torch.randn(self.nnodes, requires_grad=True)
    #     U = torch.zeros(self.nnodes)
    #     for i in self.top_order:
    #         if not self.DAG.parents_of(i):
    #             U[i] = noise[i]
    #             continue
    #         U[i] = self.f(U[list(self.DAG.parents_of(i))]) + noise[i]

    #     return U

    # def get_samples(self, num_samples=500, rescale=False):
    #      U = torch.stack([self.sample() for _ in range(num_samples)], axis=0)
    #      self.B = torch.zeros(num_samples, self.nnodes)
    #      if rescale:
    #          return self.rescale(U)
    #      return U
    
    # def get_obs_samples(self, G, num_samples=500, rescale=False):
    #     U = torch.stack([self.sample() for _ in range(num_samples)], axis=0)
    #     self.B = torch.zeros(num_samples, self.nnodes)
    #     if rescale:
    #         U = self.rescale(U)
    #     X = (G@U.T).T
    #     return U, X
    
    # def rescale(self, U):
    #     # means = torch.mean(U, dim=0)
    #     # stds = torch.std(U, dim=0)
    #     # self.A = torch.eye(U.shape[1])*torch.Tensor(1/stds)
    #     # self.B = torch.ones_like(U)*(-means/stds)
    #     # Calculate A and B
    #     for j in range(U.shape[1]):
    #         u_min = torch.min(U[:, j])
    #         u_max = torch.max(U[:, j])
    #         if u_max > u_min:
    #             self.A[j, j] = 1 / (u_max - u_min)
    #             self.B[:, j] = -self.A[j, j] * u_min
    #         else:
    #             self.A[j, j] = 1  # arbitrary non-zero to avoid division by zero
    #             self.B[:, j] = 0.5 - u_min  # shifting all values to the middle of the range
    #     # for i in range(self.nnodes):
    #     #     self.functions[i] = lambda x: torch.norm(x, p=2)**2
    #     return U@self.A+self.B
    
    # def descale(self, U):
    #     # self.A = torch.eye(self.nnodes)
    #     # self.B = torch.zeros_like(U)
    #     return (U-self.B)@torch.inverse(self.A)
    #     return (U-self.B)@torch.inverse(self.A)

    def draw_graph(self):
        print('Drawing graph...')
        draw(self.DAG)

    def get_sinks(self):
        return self.DAG.sinks()
    
    def remove_sinks(self):
        sorted_sinks = sorted(list(self.get_sinks()), reverse=True)
        for sink in sorted_sinks:
            self.DAG.remove_node(sink)
            self.removed_nodes.append(sink)
            name_map = {node:node for node in self.DAG.nodes}
            for node in range(sink+1, self.nnodes):
                name_map[node] -= 1
            self.DAG = self.DAG.rename_nodes(name_map)
            # print(name_map)
            self.nnodes-=1
            self.variances.pop(sink) #torch.cat((self.variances[:sink], self.variances[sink+1:]))
            self.means.pop(sink) #= torch.cat((self.means[:sink], self.means[sink+1:]))
            self.coeffs.pop(sink) #= torch.cat((self.coeffs[:sink], self.coeffs[sink+1:]))
            # self.A = torch.cat((self.A[:sink], self.A[sink+1:]))
            # self.A = self.A[torch.arange(self.A.size(0)) != sink][:, torch.arange(self.A.size(1)) != sink]
            # self.B = torch.cat((self.B[:sink], self.B[sink+1:]))
            # self.B = torch.cat((self.B[:,:sink], self.B[:,sink+1:]), dim=1)

        self.top_order = self.DAG.topological_sort()
        


