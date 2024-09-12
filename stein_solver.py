import torch
imp∫dort torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset
import torch.multiprocessing as mp

def Stein_hess(X, eta_G=0.01, eta_H=0.01):
    device = X.device #**
    n,d = X.shape
    
    norm = torch.cdist(X, X, p=2)
    s = torch.median(norm.flatten())
    # s = heuristic_kernel_width(X.detach())

    X_diff = X.unsqueeze(1)-X

    K = torch.exp(-norm**2 / (2 * s**2)) /s
    # K = torch.exp(-torch.cdist(X, X, p=2)**2 / (2 * s**2)) /s

    nablaK = -torch.einsum("kij,ik->kj", X_diff, K) / s**2

    G = torch.linalg.solve(K + eta_G * torch.eye(n, device=device), nablaK)

    Gv = torch.einsum("ij,ik->ijk", G, G)
    nabla2vK = torch.einsum("ikl,ikj,ik->ijl", X_diff, X_diff, K) / s**4
    diagonal_adjustment = torch.einsum("ij->i", K) / s**2
    for c in range(d):
        nabla2vK[:, c, c] -= diagonal_adjustment

    nabla2vK_flat = nabla2vK.reshape(n, -1) # shape (n, d*d)
    inv_term_flat = torch.linalg.solve(K + eta_H * torch.eye(n, device=device), nabla2vK_flat)  # shape (n, d*d)
    inv_term = inv_term_flat.reshape(n, d, d)
    inv_term = 0.5 * (inv_term + inv_term.permute(0, 2, 1))

    return -Gv + inv_term


def setup(rank, world_size):
    dist.init_process_group(
        backend='nccl',  # Use 'nccl' for NVIDIA GPUs
        init_method='tcp://127.0.0.1:29500',  # Initialize the TCP communication
        world_size=world_size,
        rank=rank
    )

class SteinHessModel(nn.Module):
    def __init__(self):
        super(SteinHessModel, self).__init__()
        self.linear = nn.Linear(10, 10)

    def forward(self, X):
        return Stein_hess(X)
    
def get_dataloader(dataset, batch_size, rank, world_size):
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler)

def main(rank, world_size):
    setup(rank, world_size)
    
    model = SteinHessModel().to(rank)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    X = torch.randn(1000, 4, requires_grad=True)  # Dummy data
    target = torch.randn(1000, 4, 4)  # Adjust target shape
    dataset = TensorDataset(X, target)
    dataloader = get_dataloader(dataset, batch_size=32, rank=rank, world_size=world_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()  # Placeholder criterion, replace with your actual loss function

    model.train()
    for epoch in range(10):
        for batch in dataloader:
            X_batch, target_batch = batch[0].to(rank), batch[1].to(rank)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, target_batch)  # Ensure shapes match
            loss.backward()
            optimizer.step()
    
    cleanup()

def cleanup():
    dist.destroy_process_group()

if __name__ == "__main__":
    world_size = 4  # Number of GPUs
    mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)