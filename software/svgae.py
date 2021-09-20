#Generating Hyperspherical variational auto-encoders
#code implemented based on https://github.com/nicola-decao/s-vae-pytorch

import torch
from torch_geometric.nn import GCNConv, InnerProductDecoder
from torch_geometric.utils import (negative_sampling, remove_self_loops, add_self_loops)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from collections import defaultdict

from hyperspherical_vae.distributions import VonMisesFisher
from hyperspherical_vae.distributions import HypersphericalUniform

from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device("cpu")

EPS = 1e-15
MAX_LOGSTD = 10


class GCNEncoder(torch.nn.Module):
    def __init__(self, i_dim, z_dim_mu, z_dim_var):
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(i_dim, 64, cached=True)
        self.conv_mu = GCNConv(64, z_dim_mu, cached=True)
        self.conv_var = GCNConv(64, z_dim_var, cached=True)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=0.1, training=self.training)
        return self.conv_mu(x, edge_index), self.conv_var(x, edge_index)




class ModelVAE(torch.nn.Module): 
    def __init__(self, i_dim, h_dim, z_dim, activation=F.relu, distribution='normal'):
        """
        ModelVAE initializer
        :param i_dim: dimension of the input data
        :param h_dim: dimension of the hidden layers
        :param z_dim: dimension of the latent representation
        :param activation: callable activation function
        :param distribution: string either `normal` or `vmf`, indicates which distribution to use
        """
        super(ModelVAE, self).__init__()
        
        self.z_dim, self.activation, self.distribution = z_dim, activation, distribution

        if self.distribution == 'normal':
            # compute mean and std of the normal distribution
            self.encoder = GCNEncoder(i_dim, z_dim, z_dim)
        elif self.distribution == 'vmf':
            # compute mean and concentration of the von Mises-Fisher
            self.encoder = GCNEncoder(i_dim, z_dim, 1)
        else:
            raise NotImplemented
            
        self.decoder = InnerProductDecoder()

    def encode(self, x, edge_index):
        # 2 hidden layers encoder
        if self.distribution == 'normal':
            # compute mean and std of the normal distribution
            z_mean, z_var = self.encoder(x, edge_index)
            z_var = F.softplus(z_var)

        elif self.distribution == 'vmf':
            # compute mean and concentration of the von Mises-Fisher
            z_mean, z_var = self.encoder(x, edge_index)
            z_mean = z_mean / z_mean.norm(dim=-1, keepdim=True)
            # the `+ 1` prevent collapsing behaviors
            z_var = F.softplus(z_var) 
        else:
            raise NotImplemented
        
        return z_mean, z_var
        
      
    def recon_loss(self, z, pos_edge_index, neg_edge_index=None):
        pos_loss = -torch.log(self.decoder(z, pos_edge_index) + EPS).mean()

        # Do not include self-loops in negative samples
        pos_edge_index, _ = remove_self_loops(pos_edge_index)
        pos_edge_index, _ = add_self_loops(pos_edge_index)
        if neg_edge_index is None:
            neg_edge_index = negative_sampling(pos_edge_index, z.size(0))
        neg_loss = -torch.log(1 -
                              self.decoder(z, neg_edge_index, sigmoid=True) +
                              EPS).mean()
        return pos_loss + neg_loss

    def reparameterize(self, z_mean, z_var):
        if self.distribution == 'normal':
            q_z = torch.distributions.normal.Normal(z_mean, z_var)
            p_z = torch.distributions.normal.Normal(torch.zeros_like(z_mean), torch.ones_like(z_var))
        elif self.distribution == 'vmf':
            # q_z = VonMisesFisher(z_mean, z_var,validate_args=False)
            # p_z = HypersphericalUniform(self.z_dim - 1,validate_args=False)
            q_z = VonMisesFisher(z_mean, z_var)
            p_z = HypersphericalUniform(self.z_dim - 1)
        else:
            raise NotImplemented

        return q_z, p_z
        
    def forward(self, x, edge_index): 
        z_mean, z_var = self.encode(x,edge_index)
        q_z, p_z = self.reparameterize(z_mean, z_var)
        z = q_z.rsample()
        return (q_z, p_z), z
    
def train(model, optimizer, x, edge_index):
   

    return loss.item()

def compute_scores(z,test_pos,test_neg):
    test = torch.cat((test_pos, test_neg),dim=1)
    labels = torch.zeros(test.size(1),1)
    labels[0:test_pos.size(1)] = 1
    row, col = test
    src = z[row]
    tgt = z[col]
    scores = torch.sigmoid(torch.sum(src * tgt,dim=1))
    auc = roc_auc_score(labels, scores)
    ap = average_precision_score(labels, scores)
    return auc,ap


def CalSVGAE(edge_index,x,test_and_val,args):
    print('___Calculating SVGAE embbeding___')
    test_pos,test_neg,val_pos,val_neg=test_and_val
    H_DIM = 16
    #Z_DIM = args.embedding_dim
    Z_DIM = 64
    num_features = x.size(1)
    num_nodes = torch.max(edge_index)
    edge_index = edge_index.to(device)
    x = x.to(device)
    distribution = 'vmf'
    if distribution == 'normal':
        model = ModelVAE(i_dim=num_features, h_dim=H_DIM, z_dim=Z_DIM, distribution='normal').to(device)
    if distribution == 'vmf':
        model = ModelVAE(i_dim=num_features, h_dim=H_DIM, z_dim=Z_DIM+1, distribution='vmf').to(device)



    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    best_val_auc=0
    for epoch in range(1, 2000 + 1):
        model.train()
        optimizer.zero_grad()
        (q_z, p_z), z = model(x, edge_index)
        loss_recon = model.recon_loss(z, edge_index)
        if model.distribution == 'normal':
            loss_KL = torch.distributions.kl.kl_divergence(q_z, p_z).sum(-1).mean()
        elif model.distribution == 'vmf':
            loss_KL = torch.distributions.kl.kl_divergence(q_z, p_z).mean()
        else:
            raise NotImplemented

        loss = loss_recon + (1 / num_nodes) * loss_KL
        loss.backward()
        optimizer.step()
        if epoch%10 == 0:
            model.eval()
            _, z  = model(x, edge_index)
            z = z.cpu().clone().detach()
            auc,_=compute_scores(z,val_pos,val_neg)
            if auc>best_val_auc:
                best_val_auc=auc
                record_z=z.clone().detach()
            print(f'Setp: {epoch:03d} /2000, Loss : {loss.item():.4f}, Val_auc:{best_val_auc:.4f}')
            
    auc,ap=compute_scores(record_z,test_pos,test_neg)
    print(f'svgae prediction accuracy, AUC: {auc:.4f}, AP: {ap:.4f}')
    del model
    torch.cuda.empty_cache()
    return record_z,auc, ap
