import numpy as np
import matplotlib.pyplot as plt
import torch
from data.generate import gen_dag
from data.visualize import draw
from random import random


class SyntheticDAG:
    def __init__(self, nnodes, DAG_type, f=lambda x: torch.norm(x, p=2)**2, seed=1234):
        np.random.seed(seed)
        self.nnodes = nnodes
        self.f = f
        self.coeffs = [1 for _ in range(nnodes)]
        self.means = [0 for _ in range(nnodes)] 
        self.variances = [1 for _ in range(nnodes)]
        self.DAG = gen_dag(nnodes, DAG_type)
        self.top_order = self.DAG.topological_sort()
        self.removed_nodes = []


    def apply_f(self,x):
        return torch.tensor([self.f(row) for row in x])

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
            self.nnodes-=1
            self.variances.pop(sink)
            self.means.pop(sink)
            self.coeffs.pop(sink) #= torch.cat((self.coeffs[:sink], self.coeffs[sink+1:]))

        self.top_order = self.DAG.topological_sort()
        


