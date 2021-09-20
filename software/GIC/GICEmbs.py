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
from gic import GIC
from logreg import LogReg
import process
from torch_geometric.utils import to_scipy_sparse_matrix, to_networkx
from torch_geometric.data import Data
import torch
import networkx as nx
import random
import string
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

cuda0 = torch.cuda.is_available()#False
# training params
batch_size = 1
nb_epochs = 2001
patience = 100
lr = 0.001
l2_coef = 0.0
drop_prob = 0.0
hid_units = 16
sparse = True
nonlinearity = 'prelu' # special name to separate parameters





cuda0 = torch.cuda.is_available()
torch.cuda.empty_cache()

def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def get_roc_score(edges_pos, edges_neg, embeddings):
    "from https://github.com/tkipf/gae"
    
    score_matrix = np.dot(embeddings, embeddings.T)
    
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    # Store positive edge predictions, actual values
    preds_pos = []
    pos = []
    for edge in edges_pos:
        preds_pos.append(sigmoid(score_matrix[edge[0], edge[1]])) # predicted score
        pos.append(1) # actual value (1 for positive)
        
    # Store negative edge predictions, actual values
    preds_neg = []
    neg = []
    for edge in edges_neg:
        preds_neg.append(sigmoid(score_matrix[edge[0], edge[1]])) # predicted score
        neg.append(0) # actual value (0 for negative)
        
    # Calculate scores
    preds_all = np.hstack([preds_pos, preds_neg])
    labels_all = np.hstack([np.ones(len(preds_pos)), np.zeros(len(preds_neg))])
    
    #print(preds_all, labels_all )
    
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)
    return roc_score, ap_score

def CalGIC(edge_index, features, dataset, test_and_val, args):
    print('Calculating GIC embbeding...')
    set_random_seed(args.seed)
    beta = 100
    num_clusters = int(10)
    alpha = 0.5
    if args.data_name=='cora':
        beta=100
        alpha=0.5
        num_clusters = int(128)
    if args.data_name=='citeseer':
        beta=100
        alpha=0.5
        num_clusters = int(128)
    if args.data_name=='pubmed':
        beta=10
        alpha=0.75
        num_clusters = int(32)




    nb_nodes = features.size(0)
    ft_size = features.size(1)
    test_pos,test_neg,val_pos,val_neg=test_and_val
    val_pos = torch.transpose(val_pos,0,1).cpu().detach().numpy()
    val_neg = torch.transpose(val_neg,0,1).cpu().detach().numpy()
    test_pos = torch.transpose(test_pos,0,1).cpu().detach().numpy()
    test_neg = torch.transpose(test_neg,0,1).cpu().detach().numpy()
    features = torch.reshape(features,(1,features.size(0),features.size(1)))
    sp_adj = to_scipy_sparse_matrix(edge_index)
    # data = Data(edge_index=edge_index)
    # g = to_networkx(data)
    # sp_adj = nx.adjacency_matrix(g)

    sp_adj = process.normalize_adj(sp_adj + sp.eye(sp_adj.shape[0]))

    sp_adj = process.sparse_mx_to_torch_sparse_tensor(sp_adj)
    if cuda0:
        features = features.cuda()
        sp_adj = sp_adj.cuda()
       
    b_xent = nn.BCEWithLogitsLoss()
    b_bce = nn.BCELoss()
    model = GIC(nb_nodes,ft_size, args.embedding_dim, nonlinearity, num_clusters, beta)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_coef)
    cnt_wait = 0
    best = 1e9
    best_t = 0
    val_best = 0
    if cuda0:
        model.cuda()
    tx=''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(10))
    dataname=dataset+tx+'-link.pkl'
    best_val_auc=0
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
            torch.save(model.state_dict(), dataname)
        else:
            cnt_wait += 1
            if cnt_wait == patience:       
                break
            loss.backward()
            optimiser.step()
        if epoch%20==1:
            model.eval()
            embeds, _,_, S= model.embed(features, sp_adj if sparse else adj, sparse, None, beta)
            embs = embeds[0, :]
            embs = embs / embs.norm(dim=1)[:, None]
            embs = embs.cpu().clone().detach()


            sc_roc, sc_ap = get_roc_score(val_pos, val_neg, embs)
            if best_val_auc<sc_roc:
                best_val_auc=sc_roc
                record_embs=embs.cpu().clone().detach()
            print(f'Setp: {epoch:03d} /2000, Loss : {loss.item():.4f}, Val_auc:{best_val_auc:.4f},cnt_wait:{cnt_wait:.4f}')


    

    sc_roc, sc_ap = get_roc_score(test_pos, test_neg, record_embs)
    print(f'GIC prediction accuracy, AUC: {sc_roc:.4f}, AP: {sc_ap:.4f}')
    print("")
    os.remove(dataname)
    return embs, sc_roc, sc_ap



