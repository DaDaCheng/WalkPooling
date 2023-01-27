from torch_geometric.utils import to_dense_adj,is_undirected, to_undirected
import numpy as np
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
def maybe_num_nodes(index, num_nodes=None):
    return index.max().item() + 1 if num_nodes is None else num_nodes

def spc(edge_index,args):
    set_random_seed(args.seed)
    if is_undirected(edge_index) == False:
        edge_index = to_undirected(edge_index)
    A=to_dense_adj(edge_index)[0]
    num_nodes=maybe_num_nodes(edge_index)
    w, v = torch.linalg.eigh(A)
    sorted, indices = torch.sort(w)
    dim=args.embedding_dim
    indices=indices[-dim:]
    #indices=indices[:dim]
    v=v[:,indices]
    out=v.cpu().detach().clone()
    return out