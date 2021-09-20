from torch_geometric.nn import Node2Vec
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def CalN2V(edge_index,args):
    modeljure = Node2Vec(edge_index, embedding_dim=args.embedding_dim, walk_length=20,
                     context_size=10, walks_per_node=10,
                     num_negative_samples=1, p=1, q=1, sparse=True).to(device)
    loader = modeljure.loader(batch_size=32, shuffle=True)
    optimizer = torch.optim.SparseAdam(list(modeljure.parameters()), lr=0.01)
    modeljure.train()
    total_loss = 0
    print('___Calculating Node2Vec features___')
    for i in range(201):
        total_loss=0
        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()
            loss = modeljure.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if i%20 == 0:
            print(f'Setp: {i:03d} /200, Loss : {total_loss:.4f}')
    output=(modeljure.forward()).cpu().clone().detach()
    del modeljure
    del loader
    torch.cuda.empty_cache()
    return output