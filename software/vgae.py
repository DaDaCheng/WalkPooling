"Implementation based on https://github.com/rusty1s/pytorch_geometric/blob/master/examples/autoencoder.py"

import torch
from torch_geometric.nn import VGAE, GCNConv
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class VariationalGCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VariationalGCNEncoder, self).__init__()
        hidden_channels=64
        self.conv1 = GCNConv(in_channels, hidden_channels, cached=True)
        self.conv_mu = GCNConv(hidden_channels, out_channels, cached=True)
        self.conv_logstd = GCNConv(hidden_channels, out_channels, cached=True)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)

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
    
def CalVGAE(edge_index, x, test_and_val, args):
    print('___Calculating VGAE embbeding___')
    test_pos,test_neg,val_pos,val_neg=test_and_val
    out_channels = int(args.embedding_dim)
    num_features = x.size(1)
    model = VGAE(VariationalGCNEncoder(num_features, out_channels)).to(device)
    edge_index = edge_index.to(device)
    x = x.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    num_nodes = torch.max(edge_index)
    best_val_auc=0
    for epoch in range(1, 500 + 1):
        model.train()
        optimizer.zero_grad()
        z = model.encode(x, edge_index)
        loss = model.recon_loss(z, edge_index)
        loss = loss + (1 / num_nodes) * model.kl_loss()
        loss.backward()
        optimizer.step()
        if epoch%10 == 0:
            model.eval()
            z = model.encode(x, edge_index)
            z = z.cpu().clone().detach()
            auc,_=compute_scores(z,val_pos,val_neg)
            if auc>best_val_auc:
                best_val_auc=auc
                record_z=z.clone().detach()
            print(f'Setp: {epoch:03d} /500, Loss : {loss.item():.4f}, Val_auc:{best_val_auc:.4f}')
            
    auc,ap=compute_scores(record_z,test_pos,test_neg)
    
    print(f'vgae prediction accuracy, AUC: {auc:.4f}, AP: {ap:.4f}')
    del model
    torch.cuda.empty_cache()
    return record_z, auc, ap
