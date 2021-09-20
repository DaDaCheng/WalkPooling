"Implementation based on https://github.com/cmavro/Graph-InfoClust-GIC"

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from torch_geometric.datasets import Planetoid
import os, sys
cur_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append('%s/models' % cur_dir)
sys.path.append('%s/utils/' % cur_dir)
print('aaa')
print(sys.path)
from gic import GIC
from logreg import LogReg
import process
from torch_geometric.utils import to_scipy_sparse_matrix, add_self_loops
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

torch.manual_seed(1234)

import statistics 
import argparse

def get_roc_score(edges_pos, edges_neg, embeddings, adj_sparse):
    "from https://github.com/tkipf/gae"
    
    score_matrix = np.dot(embeddings, embeddings.T)
    
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    # Store positive edge predictions, actual values
    preds_pos = []
    pos = []
    for edge in edges_pos:
        preds_pos.append(sigmoid(score_matrix[edge[0], edge[1]])) # predicted score
        pos.append(adj_sparse[edge[0], edge[1]]) # actual value (1 for positive)
        
    # Store negative edge predictions, actual values
    preds_neg = []
    neg = []
    for edge in edges_neg:
        preds_neg.append(sigmoid(score_matrix[edge[0], edge[1]])) # predicted score
        neg.append(adj_sparse[edge[0], edge[1]]) # actual value (0 for negative)
        
    # Calculate scores
    preds_all = np.hstack([preds_pos, preds_neg])
    labels_all = np.hstack([np.ones(len(preds_pos)), np.zeros(len(preds_neg))])
    
    #print(preds_all, labels_all )
    
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)
    return roc_score, ap_score

cuda0 = torch.cuda.is_available()#False
# training params
batch_size = 1
nb_epochs = 2000
patience = 50
lr = 0.001
l2_coef = 0.0
drop_prob = 0.0
hid_units = 16
sparse = True
nonlinearity = 'prelu' # special name to separate parameters

beta = 100
num_clusters = int(128)
alpha = 0.5

cuda0 = torch.cuda.is_available()
torch.cuda.empty_cache()

def CalGIC(edge_index,features,dataset):
    nb_nodes = features.size(0)
    ft_size = features.size(1)
    features = torch.reshape(features,(1,features.size(0),features.size(1)))
    sp_adj = to_scipy_sparse_matrix(edge_index)
    sp_adj = process.normalize_adj(sp_adj + sp.eye(sp_adj.shape[0]))

    sp_adj = process.sparse_mx_to_torch_sparse_tensor(sp_adj)
    if cuda0:
        features = features.cuda()
        sp_adj = sp_adj.cuda()
       
    b_xent = nn.BCEWithLogitsLoss()
    b_bce = nn.BCELoss()
    model = GIC(nb_nodes,ft_size, hid_units, nonlinearity, num_clusters, beta)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_coef)
    cnt_wait = 0
    best = 1e9
    best_t = 0
    val_best = 0
    if cuda0:
        model.cuda()
    for epoch in range(nb_epochs):
        model.train()
        optimiser.zero_grad()
        idx = np.random.permutation(nb_nodes)
        shuf_fts = features[:, idx, :]
        lbl_1 = torch.ones(batch_size, nb_nodes)
        lbl_2 = torch.zeros(batch_size, nb_nodes)
        lbl = torch.cat((lbl_1, lbl_2), 1)
        if cuda0:
            shuf_fts = shuf_fts.cuda()
            lbl = lbl.cuda()
        logits, logits2  = model(features, shuf_fts, sp_adj, sparse, None, None, None, beta) 
        loss = alpha* b_xent(logits, lbl)  + (1-alpha)*b_xent(logits2, lbl) 
        if loss < best:
            best = loss
            best_t = epoch
            cnt_wait = 0
            torch.save(model.state_dict(), dataset+'-link.pkl')                
        else:
            cnt_wait += 1
            if cnt_wait == patience:       
                break
            loss.backward()
            optimiser.step()
    model.load_state_dict(torch.load(dataset+'-link.pkl'))

    embeds, _,_, S= model.embed(features, sp_adj if sparse else adj, sparse, None, beta)
    embs = embeds[0, :]
    embs = embs / embs.norm(dim=1)[:, None]

    return embs.cpu().clone().detach()


def split_data(data,test_ratio,val_ratio):
    
    n_t = int(math.floor(test_ratio * data.num_edges/2)) #number of test positive edges
    n_v = int(math.floor(val_ratio * data.num_edges/2)) #number of validation positive edges

    row, col = data.edge_index
    mask = row < col
    row, col = row[mask], col[mask]
    perm = torch.randperm(row.size(0))
    row, col = row[perm], col[perm]
    r, c = row[:n_v], col[:n_v]
    data.val_pos = torch.stack([r, c], dim=0)
    r, c = row[n_v:n_v+n_t], col[n_v:n_v+n_t]
    data.test_pos = torch.stack([r, c], dim=0)
    r, c = row[n_v+n_t:], col[n_v+n_t:]
    data.train_pos = torch.stack([r, c], dim=0)

    edge_index_tp,_ = add_self_loops(data.edge_index)
    neg_edge_index = negative_sampling(edge_index_tp, num_nodes=data.num_nodes,\
            force_undirected=True)
    row, col = neg_edge_index
    mask = row<col
    row, col = row[mask], col[mask]
    perm = torch.randperm(row.size(0))
    row, col = row[perm], col[perm]
    data.val_neg = torch.stack([row[:n_v], col[:n_v]], dim=0)
    data.test_neg = torch.stack([row[n_v:n_v+n_t], col[n_v:n_v+n_t]], dim=0)
    data.train_neg = torch.stack([row[n_v+n_t:n_v+n_t+data.train_pos.size(1)],\
        col[n_v+n_t:n_v+n_t+data.train_pos.size(1)]], dim=0)
    return data

from torch_geometric.data import Data, DataLoader
from torch_geometric.transforms import NormalizeFeatures
import torch
import random
import math
from torch_geometric.utils import negative_sampling, to_networkx, is_undirected
import networkx as nx
dataset = Planetoid(root='data/', name='citeseer', transform=NormalizeFeatures())
data = dataset[0]

data = split_data(data, .1, .05)

edge_index = torch.cat((data.train_pos,data.train_pos[[1,0],:]),dim=1)
data_observed = Data(edge_index=edge_index)

x = CalGIC(data_observed.edge_index,data.x,'citeseer')

test_edges = torch.transpose(data.test_pos,0,1).cpu().detach().numpy()
test_edges_false = torch.transpose(data.test_neg,0,1).cpu().detach().numpy()
g = to_networkx(data)
adj_sparse = nx.adjacency_matrix(g)
sc_roc, sc_ap = get_roc_score(test_edges, test_edges_false, x.cpu().detach().numpy(), adj_sparse)
               

print('AUC', sc_roc, 'AP', sc_ap)



