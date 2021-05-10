import numpy as np
import lib.python.nearest_neighbors as nearest_neighbors
import time
import torch


def np_test():
    batch_size = 16
    num_points = 81920
    K = 16
    pc = np.random.rand(batch_size, num_points, 3).astype(np.float32)
    pc_query = np.random.rand(batch_size, num_points // 4, 3).astype(np.float32)

    # nearest neighbours
    start = time.time()
    neigh_idx = nearest_neighbors.knn_batch(pc, pc_query, K, omp=True)
    print(time.time() - start)


def torch_test():
    batch_size = 16
    num_points = 16384
    K = 16
    pc = torch.rand(batch_size, num_points, 3, dtype=torch.float32)
    pc_query = torch.rand(batch_size, num_points // 4, 3, dtype=torch.float32)

    # nearest neighbours
    start = time.time()
    neigh_idx = nearest_neighbors.knn_batch(pc, pc_query, K, omp=True)
    print(time.time() - start)
    print(neigh_idx.shape)


def shape_test():
    batch_size = 16
    num_points = 16384
    pc = torch.rand(batch_size, num_points, 3, dtype=torch.float32)
    bs, npoint, channel = list(pc.shape)

    print(bs)
    print(npoint)
    print(channel)


if __name__ == '__main__':
    torch_test()
    # shape_test()
