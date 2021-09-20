"Implementation based on https://github.com/PetarV-/DGI"
import torch
import torch.nn as nn
from layers import GCN, AvgReadout, Discriminator, Discriminator_cluster, Clusterator
import torch.nn.functional as F
import numpy as np



class GIC(nn.Module):
    def __init__(self,n_nb, n_in, n_h, activation, num_clusters, beta):
        super(GIC, self).__init__()
        self.gcn = GCN(n_in, n_h, activation)
        
        self.read = AvgReadout()

        self.sigm = nn.Sigmoid()

        self.disc = Discriminator(n_h)
        self.disc_c = Discriminator_cluster(n_h,n_h,n_nb,num_clusters)
        
        
        self.beta = beta
        
        self.cluster = Clusterator(n_h,num_clusters)
        
        

    def forward(self, seq1, seq2, adj, sparse, msk, samp_bias1, samp_bias2, cluster_temp):
        h_1 = self.gcn(seq1, adj, sparse)
        h_2 = self.gcn(seq2, adj, sparse)

        self.beta = cluster_temp
        
        Z, S = self.cluster(h_1[-1,:,:], cluster_temp)
        Z_t = S @ Z
        c2 = Z_t
        
        c2 = self.sigm(c2)
        
        c = self.read(h_1, msk)
        c = self.sigm(c) 
        c_x = c.unsqueeze(1)
        c_x = c_x.expand_as(h_1)
        
        ret = self.disc(c_x, h_1, h_2, samp_bias1, samp_bias2)
        
        
        ret2 = self.disc_c(c2, c2,h_1[-1,:,:], h_1[-1,:,:] ,h_2[-1,:,:], S , samp_bias1, samp_bias2)
        
        
        return ret, ret2 

    # Detach the return variables
    def embed(self, seq, adj, sparse, msk, cluster_temp):
        h_1 = self.gcn(seq, adj, sparse)
        c = self.read(h_1, msk)
        
        
        Z, S = self.cluster(h_1[-1,:,:], self.beta)
        H = S@Z
        
        
        return h_1.detach(), H.detach(), c.detach(), Z.detach()

