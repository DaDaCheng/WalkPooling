from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch
import argparse
import numpy as np
import math
from torch_geometric.utils import to_undirected, from_scipy_sparse_matrix,dense_to_sparse,is_undirected
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.datasets import Planetoid
import torch.nn.functional as F
import sys
import os.path
import pickle as pkl
import networkx as nx
import scipy.sparse as sp


cur_dir = os.path.dirname(os.path.realpath(__file__))
par_dir = os.path.abspath(os.path.join(os.path.dirname(__file__),".."))
sys.path.append('%s/software/' % par_dir)
from drnl import drnl_node_labeling

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def floor(x):
    return torch.div(x, 1, rounding_mode='trunc')

def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def split_edges(data,args):
    set_random_seed(args.seed)
    row, col = data.edge_index
    mask = row < col
    row, col = row[mask], col[mask]
    n_v= floor(args.val_ratio * row.size(0)).int() #number of validation positive edges
    n_t=floor(args.test_ratio * row.size(0)).int() #number of test positive edges
    #split positive edges   
    perm = torch.randperm(row.size(0))
    row, col = row[perm], col[perm]
    r, c = row[:n_v], col[:n_v]
    data.val_pos = torch.stack([r, c], dim=0)
    r, c = row[n_v:n_v+n_t], col[n_v:n_v+n_t]
    data.test_pos = torch.stack([r, c], dim=0)
    r, c = row[n_v+n_t:], col[n_v+n_t:]
    data.train_pos = torch.stack([r, c], dim=0)

    #sample negative edges
    if args.practical_neg_sample == False:
        # If practical_neg_sample == False, the sampled negative edges
        # in the training and validation set aware the test set

        neg_adj_mask = torch.ones(data.num_nodes, data.num_nodes, dtype=torch.uint8)
        neg_adj_mask = neg_adj_mask.triu(diagonal=1).to(torch.bool)
        neg_adj_mask[row, col] = 0

        # Sample all the negative edges and split into val, test, train negs
        neg_row, neg_col = neg_adj_mask.nonzero(as_tuple=False).t()
        perm = torch.randperm(neg_row.size(0))[:row.size(0)]
        neg_row, neg_col = neg_row[perm], neg_col[perm]

        row, col = neg_row[:n_v], neg_col[:n_v]
        data.val_neg = torch.stack([row, col], dim=0)

        row, col = neg_row[n_v:n_v + n_t], neg_col[n_v:n_v + n_t]
        data.test_neg = torch.stack([row, col], dim=0)

        row, col = neg_row[n_v + n_t:], neg_col[n_v + n_t:]
        data.train_neg = torch.stack([row, col], dim=0)

    else:

        neg_adj_mask = torch.ones(data.num_nodes, data.num_nodes, dtype=torch.uint8)
        neg_adj_mask = neg_adj_mask.triu(diagonal=1).to(torch.bool)
        neg_adj_mask[row, col] = 0

        # Sample the test negative edges first
        neg_row, neg_col = neg_adj_mask.nonzero(as_tuple=False).t()
        perm = torch.randperm(neg_row.size(0))[:n_t]
        neg_row, neg_col = neg_row[perm], neg_col[perm]
        data.test_neg = torch.stack([neg_row, neg_col], dim=0)

        # Sample the train and val negative edges with only knowing 
        # the train positive edges
        row, col = data.train_pos
        neg_adj_mask = torch.ones(data.num_nodes, data.num_nodes, dtype=torch.uint8)
        neg_adj_mask = neg_adj_mask.triu(diagonal=1).to(torch.bool)
        neg_adj_mask[row, col] = 0

        # Sample the train and validation negative edges
        neg_row, neg_col = neg_adj_mask.nonzero(as_tuple=False).t()

        n_tot = n_v + data.train_pos.size(1)
        perm = torch.randperm(neg_row.size(0))[:n_tot]
        neg_row, neg_col = neg_row[perm], neg_col[perm]

        row, col = neg_row[:n_v], neg_col[:n_v]
        data.val_neg = torch.stack([row, col], dim=0)

        row, col = neg_row[n_v:], neg_col[n_v:]
        data.train_neg = torch.stack([row, col], dim=0)

    return data


def k_hop_subgraph(node_idx, num_hops, edge_index, max_nodes_per_hop = None,num_nodes = None):
  
    if num_nodes == None:
        num_nodes = torch.max(edge_index)+1
    row, col = edge_index
    node_mask = row.new_empty(num_nodes, dtype=torch.bool)
    edge_mask = row.new_empty(row.size(0), dtype=torch.bool)

    node_idx = node_idx.to(row.device)

    subsets = [node_idx]

    if max_nodes_per_hop == None:
        for _ in range(num_hops):
            node_mask.fill_(False)
            node_mask[subsets[-1]] = True
            torch.index_select(node_mask, 0, row, out = edge_mask)
            subsets.append(col[edge_mask])
    else:
        not_visited = row.new_empty(num_nodes, dtype=torch.bool)
        not_visited.fill_(True)
        for _ in range(num_hops):
            node_mask.fill_(False)# the source node mask in this hop
            node_mask[subsets[-1]] = True #mark the sources
            not_visited[subsets[-1]] = False # mark visited nodes
            torch.index_select(node_mask, 0, row, out = edge_mask) # indices of all neighbors
            neighbors = col[edge_mask].unique() #remove repeats
            neighbor_mask = row.new_empty(num_nodes, dtype=torch.bool) # mask of all neighbor nodes
            edge_mask_hop = row.new_empty(row.size(0), dtype=torch.bool) # selected neighbor mask in this hop
            neighbor_mask.fill_(False)
            neighbor_mask[neighbors] = True
            neighbor_mask = torch.logical_and(neighbor_mask, not_visited) # all neighbors that are not visited
            ind = torch.where(neighbor_mask==True) #indicies of all the unvisited neighbors
            if ind[0].size(0) > max_nodes_per_hop:
                perm = torch.randperm(ind[0].size(0))
                ind = ind[0][perm]
                neighbor_mask[ind[max_nodes_per_hop:]] = False # randomly select max_nodes_per_hop nodes
                torch.index_select(neighbor_mask, 0, col, out = edge_mask_hop)# find the indicies of selected nodes
                edge_mask = torch.logical_and(edge_mask,edge_mask_hop) # change edge_mask
            subsets.append(col[edge_mask])

    subset, inv = torch.cat(subsets).unique(return_inverse=True)
    inv = inv[:node_idx.numel()]

    node_mask.fill_(False)
    node_mask[subset] = True
    edge_mask = node_mask[row] & node_mask[col]

    edge_index = edge_index[:, edge_mask]

    node_idx = row.new_full((num_nodes, ), -1)
    node_idx[subset] = torch.arange(subset.size(0), device=row.device)
    edge_index = node_idx[edge_index]

    return subset, edge_index, inv, edge_mask

def plus_edge(data_observed, label, p_edge, args):
    nodes, edge_index_m, mapping, _ = k_hop_subgraph(node_idx= p_edge, num_hops=args.num_hops,\
 edge_index = data_observed.edge_index, max_nodes_per_hop=args.max_nodes_per_hop ,num_nodes=data_observed.num_nodes)
    x_sub = data_observed.x[nodes,:]
    edge_index_p = edge_index_m
    edge_index_p = torch.cat((edge_index_p, mapping.view(-1,1)),dim=1)
    edge_index_p = torch.cat((edge_index_p, mapping[[1,0]].view(-1,1)),dim=1)

    #edge_mask marks the edge under perturbation, i.e., the candidate edge for LP
    edge_mask = torch.ones(edge_index_p.size(1),dtype=torch.bool)
    edge_mask[-1] = False
    edge_mask[-2] = False

    if args.drnl == True:
        num_nodes = torch.max(edge_index_p)+1
        z = drnl_node_labeling(edge_index_m, mapping[0],mapping[1],num_nodes)
        data = Data(edge_index = edge_index_p, x = x_sub, z = z)
    else:
        data = Data(edge_index = edge_index_p, x = x_sub, z = 0)
    data.edge_mask = edge_mask

    #label = 1 if the candidate link (p_edge) is positive and label=0 otherwise
    data.label = float(label)

    return data

def minus_edge(data_observed, label, p_edge, args):
    nodes, edge_index_p, mapping,_ = k_hop_subgraph(node_idx= p_edge, num_hops=args.num_hops,\
 edge_index = data_observed.edge_index,max_nodes_per_hop=args.max_nodes_per_hop, num_nodes = data_observed.num_nodes)
    x_sub = data_observed.x[nodes,:]

    #edge_mask marks the edge under perturbation, i.e., the candidate edge for LP
    edge_mask = torch.ones(edge_index_p.size(1), dtype = torch.bool)
    ind = torch.where((edge_index_p == mapping.view(-1,1)).all(dim=0))
    edge_mask[ind[0]] = False
    ind = torch.where((edge_index_p == mapping[[1,0]].view(-1,1)).all(dim=0))
    edge_mask[ind[0]] = False
    if args.drnl == True:
        num_nodes = torch.max(edge_index_p)+1
        z = drnl_node_labeling(edge_index_p[:,edge_mask], mapping[0],mapping[1],num_nodes)
        data = Data(edge_index = edge_index_p, x= x_sub,z = z)
    else:
        data = Data(edge_index = edge_index_p, x= x_sub,z = 0)
    data.edge_mask = edge_mask

    #label = 1 if the candidate link (p_edge) is positive and label=0 otherwise
    data.label = float(label)
    return data


def load_splitted_data(args):
    par_dir = os.path.abspath(os.path.join(os.path.dirname(__file__),".."))
    data_name=args.data_name+'_split_'+args.data_split_num
    if args.test_ratio==0.5:
        data_dir = os.path.join(par_dir, 'data/splitted_0_5/{}.mat'.format(data_name))
    else:
        data_dir = os.path.join(par_dir, 'data/splitted/{}.mat'.format(data_name))
    import scipy.io as sio
    print('Load data from: '+data_dir)
    net = sio.loadmat(data_dir)
    data = Data()

    data.train_pos = torch.from_numpy(np.int64(net['train_pos']))
    data.train_neg = torch.from_numpy(np.int64(net['train_neg']))
    data.test_pos = torch.from_numpy(np.int64(net['test_pos']))
    data.test_neg = torch.from_numpy(np.int64(net['test_neg']))

    n_pos= floor(args.val_ratio * len(data.train_pos)).int()
    nlist=np.arange(len(data.train_pos))
    np.random.shuffle(nlist)
    val_pos_list=nlist[:n_pos]
    train_pos_list=nlist[n_pos:]
    data.val_pos=data.train_pos[val_pos_list]
    data.train_pos=data.train_pos[train_pos_list]

    n_neg = floor(args.val_ratio * len(data.train_neg)).int()
    nlist=np.arange(len(data.train_neg))
    np.random.shuffle(nlist)
    val_neg_list=nlist[:n_neg]
    train_neg_list=nlist[n_neg:]
    data.val_neg=data.train_neg[val_neg_list]
    data.train_neg=data.train_neg[train_neg_list]

    data.val_pos = torch.transpose(data.val_pos,0,1)
    data.val_neg = torch.transpose(data.val_neg,0,1)
    data.train_pos = torch.transpose(data.train_pos,0,1)
    data.train_neg = torch.transpose(data.train_neg,0,1)
    data.test_pos = torch.transpose(data.test_pos,0,1)
    data.test_neg = torch.transpose(data.test_neg,0,1)
    num_nodes = max(torch.max(data.train_pos), torch.max(data.test_pos))+1
    num_nodes=max(num_nodes,torch.max(data.val_pos)+1)
    data.num_nodes = num_nodes

    return data

def load_unsplitted_data(args):
    # read .mat format files
    data_dir = os.path.join(par_dir, 'data/{}.mat'.format(args.data_name))
    print('Load data from: '+ data_dir)
    import scipy.io as sio
    net = sio.loadmat(data_dir)
    edge_index,_ = from_scipy_sparse_matrix(net['net'])
    data = Data(edge_index=edge_index)
    if is_undirected(data.edge_index) == False: #in case the dataset is directed
        data.edge_index = to_undirected(data.edge_index)
    data.num_nodes = torch.max(data.edge_index)+1
    return data
def load_Planetoid_data(args):
    print('Using data: '+ args.data_name)
    #dataset = Planetoid(root=par_dir+'/data/', name=args.data_name, transform=NormalizeFeatures())
    dataset = Planetoid(root=par_dir+'/data/', name=args.data_name)
    data = dataset[0]
    data.num_nodes = torch.max(data.edge_index)+1
    return data
# def load_Planetoid_data(args):
#     print('downloading data: '+ args.data_name)
#     #dataset = Planetoid(root=par_dir+'/data/', name=args.data_name, transform=NormalizeFeatures())
#     dataset = Planetoid(root=par_dir+'/data/', name=args.data_name)
#     # Edited from https://github.com/tkipf/gae/blob/master/gae/input_data.py
#     names = ['x', 'tx', 'allx', 'graph']
#     objects = []
#     for i in range(len(names)):
#         with open("./data/{}/raw/ind.{}.{}".format(args.data_name,args.data_name, names[i]), 'rb') as f:
#             if sys.version_info > (3, 0):
#                 objects.append(pkl.load(f, encoding='latin1'))
#             else:
#                 objects.append(pkl.load(f))
#     x, tx, allx, graph = tuple(objects)
#     filename="./data/{}/raw/ind.{}.test.index".format(args.data_name,args.data_name)
#     index = []
#     for line in open(filename):
#         index.append(int(line.strip()))
#     test_idx_reorder = index
#     test_idx_range = np.sort(test_idx_reorder)
#     if args.data_name == 'citeseer':
#         # Fix citeseer dataset (there are some isolated nodes in the graph)
#         # Find isolated nodes, add them as zero-vecs into the right position
#         test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
#         tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
#         tx_extended[test_idx_range-min(test_idx_range), :] = tx
#         tx = tx_extended
#     features = sp.vstack((allx, tx)).tolil()
#     features[test_idx_reorder, :] = features[test_idx_range, :]
#     features=torch.tensor(sp.coo_matrix.todense(features)).float()
#     adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
#     edge_index=from_scipy_sparse_matrix(adj)[0]
#     data=Data(edge_index=edge_index,x=features)
#     data.num_nodes = torch.max(data.edge_index)+1

#     return data


def set_init_attribute_representation(data,args):
    #Construct data_observed and compute its node attributes & representation
    edge_index = torch.cat((data.train_pos,data.train_pos[[1,0],:]),dim=1)
    if args.observe_val_and_injection == False:
        data_observed = Data(edge_index=edge_index)
    else:
        edge_index=torch.cat((edge_index,data.val_pos,data.val_pos[[1,0],:]),dim=1)
        data_observed = Data(edge_index=edge_index)
    data_observed.num_nodes = data.num_nodes
    if args.observe_val_and_injection == False:
        edge_index_observed = data_observed.edge_index
    else: 
        #use the injection trick and add val data in observed graph 
        edge_index_observed = torch.cat((data_observed.edge_index,\
            data.train_neg,data.train_neg[[1,0],:],data.val_neg,data.val_neg[[1,0],:]),dim=1)
    #generate the initial node attribute if there isn't any
    if data.x == None:
        if args.init_attribute =='n2v':
            from node2vec import CalN2V
            x = CalN2V(edge_index_observed,args)
        if args.init_attribute =='one_hot':
            x = F.one_hot(torch.arange(data.num_nodes), num_classes=data.num_nodes)
            x = x.float()
        if args.init_attribute =='spc':
            from SPC import spc
            x = spc(edge_index_observed,args)
            x = x.float()
        if args.init_attribute =='ones':
            x = torch.ones(data.num_nodes,args.embedding_dim)
            x = x.float()
        if args.init_attribute =='zeros':
            x = torch.zeros(data.num_nodes,args.embedding_dim)
            x = x.float()
    else:
        x = data.x
    #generate the initial node representation using unsupervised models
    if args.init_representation != None:
        val_and_test=[data.test_pos,data.test_neg,data.val_pos,data.val_neg]
        num_nodes,_=x.shape
        #add self-loop for the last node to aviod losing node if the last node dosen't have a link.
        if (num_nodes-1) in edge_index_observed:
            edge_index_observed=edge_index_observed.clone().detach()
        else:
            edge_index_observed=torch.cat((edge_index_observed.clone().detach(),torch.tensor([[num_nodes-1],[num_nodes-1]])),dim=1)
        if args.init_representation == 'gic':
            args.par_dir = os.path.abspath(os.path.join(os.path.dirname(__file__),".."))
            sys.path.append('%s/software/GIC/' % args.par_dir)
            from GICEmbs import CalGIC
            data_observed.x, auc, ap = CalGIC(edge_index_observed, x, args.data_name, val_and_test,args)

        if args.init_representation == 'vgae':
            from vgae import CalVGAE
            data_observed.x, auc, ap = CalVGAE(edge_index_observed, x, val_and_test, args)
        if args.init_representation == 'svgae':
            from svgae import CalSVGAE
            data_observed.x, auc, ap = CalSVGAE(edge_index_observed, x, val_and_test, args)
        if args.init_representation == 'argva':
            from argva import CalARGVA
            data_observed.x, auc, ap = CalARGVA(edge_index_observed, x, val_and_test, args)
        feature_results=[auc,ap]
    else:
        data_observed.x = x
        feature_results=None

    return data_observed,feature_results

def prepare_data(args):
    #load data from .mat or download from Planetoid dataset.
    
    if args.data_name in ('cora', 'citeseer', 'pubmed'):
        data = load_Planetoid_data(args)
        data = split_edges(data,args)
    elif args.data_name in ('chameleon','squirrel','film','cornell','texas','wisconsin'):
        datastr = args.data_name
        split_index=str(0)## this split is node-classification split from geom-gcn, not for link prediction
        splitstr = 'data/new_data_splits/'+datastr+'_split_0.6_0.2_'+split_index+'.npz'
        g, features, labels, _, _, _, num_features, num_labels = new_load_data(datastr,splitstr)
        A=g.toarray()
        edge_index,_=dense_to_sparse(torch.tensor(A))
        data=Data(edge_index=edge_index,x=features.to(torch.float))
        data = split_edges(data,args)
    else:
        if args.use_splitted == True: #use splitted train/val/test
            data = load_splitted_data(args)
        else:
            data = load_unsplitted_data(args)
            data = split_edges(data,args)
    
    

    set_random_seed(args.seed)
    data_observed,feature_results= set_init_attribute_representation(data,args)

    #Construct train, val and test data loader.
    set_random_seed(args.seed)
    train_graphs = []
    val_graphs = []
    test_graphs = []
    for i in range(data.train_pos.size(1)):
        train_graphs.append(minus_edge(data_observed,1,data.train_pos[:,i],args))

    for i in range(data.train_neg.size(1)):
        train_graphs.append(plus_edge(data_observed,0,data.train_neg[:,i],args))

    for i in range(data.test_pos.size(1)):
        test_graphs.append(plus_edge(data_observed,1,data.test_pos[:,i],args))

    for i in range(data.test_neg.size(1)):
        test_graphs.append(plus_edge(data_observed,0,data.test_neg[:,i],args))   
    if args.observe_val_and_injection == False:
        for i in range(data.val_pos.size(1)):
            val_graphs.append(plus_edge(data_observed,1,data.val_pos[:,i],args))

        for i in range(data.val_neg.size(1)):
            val_graphs.append(plus_edge(data_observed,0,data.val_neg[:,i],args))
    else:
        for i in range(data.val_pos.size(1)):
            val_graphs.append(minus_edge(data_observed,1,data.val_pos[:,i],args))

        for i in range(data.val_neg.size(1)):
            val_graphs.append(plus_edge(data_observed,0,data.val_neg[:,i],args))


    
    print('Train_link:', str(len(train_graphs)),' Val_link:',str(len(val_graphs)),' Test_link:',str(len(test_graphs)))

    train_loader = DataLoader(train_graphs,batch_size=args.batch_size,shuffle=True)
    val_loader = DataLoader(val_graphs,batch_size=args.batch_size,shuffle=True)
    test_loader = DataLoader(test_graphs,batch_size=args.batch_size,shuffle=False)

    return train_loader, val_loader, test_loader,feature_results


    

#adapted from geom-gcn
def new_load_data(dataset_name, splits_file_path=None):

    graph_adjacency_list_file_path = os.path.join(par_dir,'data','new_data', dataset_name, 'out1_graph_edges.txt')
    graph_node_features_and_labels_file_path = os.path.join(par_dir,'data','new_data', dataset_name,
                                                            'out1_node_feature_label.txt')

    G = nx.DiGraph()
    graph_node_features_dict = {}
    graph_labels_dict = {}

    if dataset_name=='film':
         with open(graph_node_features_and_labels_file_path) as graph_node_features_and_labels_file:
            graph_node_features_and_labels_file.readline()
            for line in graph_node_features_and_labels_file:
                line = line.rstrip().split('\t')
                assert (len(line) == 3)
                assert (int(line[0]) not in graph_node_features_dict and int(line[0]) not in graph_labels_dict)
                feature_blank = np.zeros(932, dtype=np.uint8)
                feature_blank[np.array(line[1].split(','), dtype=np.uint16)] = 1
                graph_node_features_dict[int(line[0])] = feature_blank
                graph_labels_dict[int(line[0])] = int(line[2])

    else:
        with open(graph_node_features_and_labels_file_path) as graph_node_features_and_labels_file:
            graph_node_features_and_labels_file.readline()
            for line in graph_node_features_and_labels_file:
                line = line.rstrip().split('\t')
                assert (len(line) == 3)
                assert (int(line[0]) not in graph_node_features_dict and int(line[0]) not in graph_labels_dict)
                graph_node_features_dict[int(line[0])] = np.array(line[1].split(','), dtype=np.uint8)
                graph_labels_dict[int(line[0])] = int(line[2])

    with open(graph_adjacency_list_file_path) as graph_adjacency_list_file:
        graph_adjacency_list_file.readline()
        for line in graph_adjacency_list_file:
            line = line.rstrip().split('\t')
            assert (len(line) == 2)
            if int(line[0]) not in G:
                G.add_node(int(line[0]), features=graph_node_features_dict[int(line[0])],
                           label=graph_labels_dict[int(line[0])])
            if int(line[1]) not in G:
                G.add_node(int(line[1]), features=graph_node_features_dict[int(line[1])],
                           label=graph_labels_dict[int(line[1])])
            G.add_edge(int(line[0]), int(line[1]))

    adj = nx.adjacency_matrix(G, sorted(G.nodes()))

    features = np.array(
        [features for _, features in sorted(G.nodes(data='features'), key=lambda x: x[0])])
    labels = np.array(
        [label for _, label in sorted(G.nodes(data='label'), key=lambda x: x[0])])
    features = preprocess_features(features)
    g = adj


    splits_file_path = os.path.join(par_dir,splits_file_path)  
    with np.load(splits_file_path) as splits_file:
        train_mask = splits_file['train_mask']
        val_mask = splits_file['val_mask']
        test_mask = splits_file['test_mask']
    
    num_features = features.shape[1]
    num_labels = len(np.unique(labels))
    assert (np.array_equal(np.unique(labels), np.arange(len(np.unique(labels)))))

    features = torch.FloatTensor(features)
    labels = torch.LongTensor(labels)
    train_mask = torch.BoolTensor(train_mask)
    val_mask = torch.BoolTensor(val_mask)
    test_mask = torch.BoolTensor(test_mask)

    return g, features, labels, train_mask, val_mask, test_mask, num_features, num_labels

def preprocess_features(features):
    #print(features.sum())
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    rowsum = (rowsum==0)*1+rowsum
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    #print(features.sum())
    return features



