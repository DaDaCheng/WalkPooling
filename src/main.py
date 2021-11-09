import torch
import numpy as np
import argparse
import os.path
from utils import prepare_data
from model import LinkPred
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', device)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
def str2none(v):
    if v.lower()=='none':
        return None
    else:
        return str(v)
def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)



parser = argparse.ArgumentParser(description='Link Prediction with Walk-Pooling')
#Dataset 
parser.add_argument('--data-name', default='USAir', help='graph name')

#training/validation/test divison and ratio
parser.add_argument('--use-splitted', type=str2bool, default=True,
                    help='use the pre-splitted train/test data,\
                     if False, then make a random division')
parser.add_argument('--data-split-num',type=str, default='10',
                    help='If use-splitted is true, choose one of splitted data')
parser.add_argument('--test-ratio', type=float, default=0.1,
                    help='ratio of test links')
parser.add_argument('--val-ratio', type=float, default=0.05,
                    help='ratio of validation links. If using the splitted data from SEAL,\
                     it is the ratio on the observed links, othewise, it is the ratio on the whole links.')
parser.add_argument('--practical-neg-sample', type=bool, default = False,
                    help='only see the train positive edges when sampling negative')
#setups in peparing the training set 
parser.add_argument('--observe-val-and-injection', type=str2bool, default = True,
                    help='whether to contain the validation set in the observed graph and apply injection trick')
parser.add_argument('--num-hops', type=int, default=2,
                    help='number of hops in sampling subgraph')
parser.add_argument('--max-nodes-per-hop', type=int, default=None)


#prepare initial node attributes for those graphs do not have
parser.add_argument('--init-attribute', type=str2none, default='ones',
                    help='initial attribute for graphs without node attributes\
                    , options: n2v, one_hot, spc, ones, zeros, None')

#prepare initial node representation using unsupservised models 
parser.add_argument('--init-representation', type=str2none, default= None,
                    help='options: gic, vgae, argva, None')
parser.add_argument('--embedding-dim', type=int, default= 32,
                    help='Dimension of the initial node representation, default: 32)')

#Drnl feature in the SEAL paper
parser.add_argument('--drnl', type=str2bool, default=False,
                    help='whether to use drnl labeling')


#Model and Training
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--lr', type=float, default=0.00005,
                    help='learning rate')
parser.add_argument('--weight-decay', type=float, default=0)
parser.add_argument('--walk-len', type=int, default=7, help='cutoff in the length of walks')
parser.add_argument('--heads', type=int, default=2,
                    help='using multi-heads in the attention link weight encoder ')
parser.add_argument('--hidden-channels', type=int, default=32)
parser.add_argument('--batch-size', type=int, default=32)
parser.add_argument('--epoch-num', type=int, default=50)
parser.add_argument('--MSE', type=str2bool, default=False)
parser.add_argument('--log', type=str, default=None,
                    help='log by tensorboard, default is None')

args = parser.parse_args()

if args.data_name in ('cora', 'citeseer','pubmed'):
    args.use_splitted = False
    args.practical_neg_sample = True
    args.observe_val_and_injection = False
    args.init_attribute=None

if (args.data_name in ('Ecoli','PB','pubmed')) and (args.max_nodes_per_hop==None):
    args.max_nodes_per_hop=100

if args.data_name=='Power':
    args.num_hops=3


print ("-"*50+'Dataset and Features'+"-"*60)
print ("{:<10}|{:<10}|{:<10}|{:<10}|{:<10}|{:<10}|{:<10}|{:<10}|{:<15}|{:<20}"\
    .format('Dataset','Test Ratio','Val Ratio','Split Num','Dimension',\
        'Attribute','hops','DRNL','Representation','Observe val and injection'))
print ("-"*130)
print ("{:<10}|{:<10}|{:<10}|{:<10}|{:<10}|{:<10}|{:<10}|{:<10}|{:<15}|{:<20}"\
    .format(args.data_name,args.test_ratio,args.val_ratio,\
        args.data_split_num,args.embedding_dim,str(args.init_attribute),\
        str(args.num_hops),str(args.drnl),str(args.init_representation),str(args.observe_val_and_injection)))
print ("-"*130)


print('<<Begin generating training data>>')
train_loader, val_loader, test_loader, feature_results = prepare_data(args)
print('<<Complete generating training data>>')


print ("-"*42+'Model and Training'+"-"*45)
print ("{:<13}|{:<13}|{:<13}|{:<8}|{:<13}|{:<8}|{:<15}"\
    .format('Learning Rate','Weight Decay','Batch Size','Epoch',\
        'Walk Length','Heads','Hidden Channels'))
print ("-"*105)

print ("{:<13}|{:<13}|{:<13}|{:<8}|{:<13}|{:<8}|{:<15}"\
    .format(args.lr,args.weight_decay, str(args.batch_size),\
        args.epoch_num,args.walk_len, args.heads, args.hidden_channels))
print ("-"*105)


walk_len = args.walk_len
heads = args.heads
hidden_channels=args.hidden_channels
lr=args.lr
weight_decay=args.weight_decay

torch.cuda.empty_cache()

num_features = next(iter(train_loader)).x.size(1)

z_max=0
if args.drnl==True:
    for data in train_loader:
        z_max = max(z_max, torch.max(data.z).numpy())
    for data in val_loader:
        z_max = max(z_max, torch.max(data.z).numpy())
    for data in test_loader:
        z_max = max(z_max, torch.max(data.z).numpy())
    z_max = z_max+1
    
    #if use drnl, we use a Embedding with dimension = hidden_channels
    num_features = hidden_channels + num_features

torch.cuda.empty_cache()
print("Dimention of features after concatenation:",num_features)
set_random_seed(args.seed)

model = LinkPred(in_channels = num_features, hidden_channels = hidden_channels,\
    heads = heads, walk_len = walk_len, drnl = args.drnl,z_max = z_max, MSE= args.MSE).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=lr,weight_decay=weight_decay)
criterion = torch.nn.MSELoss(reduction='mean')


if args.MSE:
    criterion = torch.nn.MSELoss(reduction='mean')
else:
    criterion = torch.nn.BCEWithLogitsLoss()

def train(loader,epoch):
    model.train()
    loss_epoch=0
    for data in tqdm(loader,desc="train"):  # Iterate in batches over the training dataset.
        data = data.to(device)
        label= data.label
        out = model(data.x, data.edge_index, data.edge_mask, data.batch, data.z)
        torch.cuda.empty_cache()
        loss = criterion(out.view(-1), label)  
        optimizer.zero_grad()
        loss.backward()  
        optimizer.step()
        loss_epoch=loss_epoch+loss.item()
    return loss_epoch/len(loader)



def test(loader,data_type='test'):
    model.eval()
    scores = torch.tensor([])
    labels = torch.tensor([])
    loss_total=0
    with torch.no_grad():
        #for data in tqdm(loader,position=0,leave=True):  # Iterate in batches over the training/test dataset.
        for data in tqdm(loader,desc='test:'+data_type):  # Iterate in batches over the training/test dataset.
            data = data.to(device)
            out = model(data.x, data.edge_index, data.edge_mask, data.batch, data.z)
            loss = criterion(out.view(-1), data.label)
            out = out.cpu().clone().detach()
            scores = torch.cat((scores,out),dim = 0)
            labels = torch.cat((labels,data.label.view(-1,1).cpu().clone().detach()),dim = 0)
        scores = scores.cpu().clone().detach().numpy()
        labels = labels.cpu().clone().detach().numpy()
        loss_total=loss_total+loss.item()
        return roc_auc_score(labels, scores), average_precision_score(labels, scores),loss_total
       


Best_Val_fromloss=1e10
Final_Test_AUC_fromloss=0
Final_Test_AP_fromloss=0

Best_Val_fromAUC=0
Final_Test_AUC_fromAUC=0
Final_Test_AP_fromAUC=0

for epoch in range(0, args.epoch_num):
    loss_epoch = train(train_loader,epoch)
    val_auc, val_ap, val_loss = test(val_loader,data_type='val')
    test_auc,test_ap,_ = test(test_loader,data_type='test')
    if val_loss < Best_Val_fromloss:
        Best_Val_fromloss = val_loss
        Final_Test_AUC_fromloss = test_auc
        Final_Test_AP_fromloss = test_ap

    if val_auc > Best_Val_fromAUC:
        Best_Val_fromAUC = val_auc
        Final_Test_AUC_fromAUC = test_auc
        Final_Test_AP_fromAUC = test_ap
    print(f'Epoch: {epoch:03d}, Loss : {loss_epoch:.4f},\
     Val Loss : {val_loss:.4f}, Val AUC: {val_auc:.4f},\
      Test AUC: {test_auc:.4f}, Picked AUC:{Final_Test_AUC_fromAUC:.4f}')

print(f'From loss: Final Test AUC: {Final_Test_AUC_fromloss:.4f}, Final Test AP: {Final_Test_AP_fromloss:.4f}')
print(f'From AUC: Final Test AUC: {Final_Test_AUC_fromAUC:.4f}, Final Test AP: {Final_Test_AP_fromAUC:.4f}')

if isinstance(args.log,str) :
    from torch.utils.tensorboard import SummaryWriter
    comment=args.data_name
    writer = SummaryWriter(comment=(comment+'_'+args.log))
    if args.use_splitted:
        record_index=int(args.data_split_num)
    else:
        record_index=args.seed
    writer.add_scalar('AUC_from_AUC', Final_Test_AUC_fromAUC, record_index)
    writer.add_scalar('AUC_from_loss', Final_Test_AUC_fromloss, record_index)
    writer.add_scalar('AP_from_AUC', Final_Test_AP_fromAUC, record_index)
    writer.add_scalar('AP_from_loss', Final_Test_AP_fromloss, record_index)
    if feature_results is not None:
        [auc,ap]=feature_results
        re_log_auc=args.init_representation+'_auc'
        re_log_ap=args.init_representation+'_ap'
        writer.add_scalar(re_log_auc, auc, record_index)
        writer.add_scalar(re_log_ap, ap, record_index)
    writer.close()
