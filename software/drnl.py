"Code adopted and implemented from https://github.com/muhanzhang/SEAL"
import scipy.sparse as ssp
from scipy.sparse.csgraph import shortest_path
import torch
import numpy as np

def drnl_node_labeling(edge_index, src, dst, num_nodes):

    edge_weight = torch.ones(edge_index.size(1), dtype=int)
    adj = ssp.csr_matrix(
            (edge_weight, (edge_index[0], edge_index[1])), 
            shape=(num_nodes, num_nodes))
    # Double Radius Node Labeling (DRNL).
    src, dst = (dst, src) if src > dst else (src, dst)

    idx = list(range(src)) + list(range(src + 1, adj.shape[0]))
    adj_wo_src = adj[idx, :][:, idx]

    idx = list(range(dst)) + list(range(dst + 1, adj.shape[0]))
    adj_wo_dst = adj[idx, :][:, idx]

    dist2src = shortest_path(adj_wo_dst, directed=False, unweighted=True, indices=src)
    dist2src = np.insert(dist2src, dst, 0, axis=0)
    dist2src = torch.from_numpy(dist2src)

    dist2dst = shortest_path(adj_wo_src, directed=False, unweighted=True, indices=dst-1)
    dist2dst = np.insert(dist2dst, src, 0, axis=0)
    dist2dst = torch.from_numpy(dist2dst)

    dist = dist2src + dist2dst
    dist_over_2, dist_mod_2 = dist // 2, dist % 2

    z = 1 + torch.min(dist2src, dist2dst)
    z += dist_over_2 * (dist_over_2 + dist_mod_2 - 1)
    z[src] = 1.
    z[dst] = 1.
    z[torch.isnan(z)] = 0.
    return z.to(torch.int)

# def drnl_node_labeling(edge_index, src, dst, num_nodes):
#     # Distance Encoding Plus. When computing distance to src, temporarily mask dst;
#     # when computing distance to dst, temporarily mask src. Essentially the same as DRNL.
#     max_dist=100
#     edge_weight = torch.ones(edge_index.size(1), dtype=int)
#     adj = ssp.csr_matrix(
#             (edge_weight, (edge_index[0], edge_index[1])), 
#             shape=(num_nodes, num_nodes))
#     src, dst = (dst, src) if src > dst else (src, dst)

#     idx = list(range(src)) + list(range(src + 1, adj.shape[0]))
#     adj_wo_src = adj[idx, :][:, idx]

#     idx = list(range(dst)) + list(range(dst + 1, adj.shape[0]))
#     adj_wo_dst = adj[idx, :][:, idx]

#     dist2src = shortest_path(adj_wo_dst, directed=False, unweighted=True, indices=src)
#     dist2src = np.insert(dist2src, dst, 0, axis=0)
#     dist2src = torch.from_numpy(dist2src)

#     dist2dst = shortest_path(adj_wo_src, directed=False, unweighted=True, indices=dst-1)
#     dist2dst = np.insert(dist2dst, src, 0, axis=0)
#     dist2dst = torch.from_numpy(dist2dst)

#     dist = torch.cat([dist2src.view(-1, 1), dist2dst.view(-1, 1)], 1)
#     dist[dist > max_dist] = max_dist
#     dist[torch.isnan(dist)] = max_dist + 1
#     dist = torch.sum(dist,dim=1)

#     return dist.to(torch.int)
