import os
import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx

from dgl import DGLGraph
from utils import accuracy, roc_auc, preprocess_data
from model import MixHopGCN  # Changed from FAGCN to MixHopGCN

import warnings
warnings.simplefilter('ignore')

def fix_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='new_squirrel')
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-5, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--epochs', type=int, default=500, help='Number of epochs to train.')
parser.add_argument('--hidden', type=int, default=32, help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
parser.add_argument('--num_hops', type=int, default=2, help='Number of hops in MixHop layer')
parser.add_argument('--combine', type=str, default='concat', choices=['concat', 'sum'], help='How to combine multi-hop outputs')
parser.add_argument('--layer_num', type=int, default=2, help='Number of layers')
parser.add_argument('--train_ratio', type=float, default=0.6, help='Ratio of training set')
parser.add_argument('--patience', type=int, default=200, help='Patience')
parser.add_argument('--remove_zero_in_degree_nodes', action='store_true')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--dataset_file_path', type=str, required=True)
parser.add_argument('--result_file_path', type=str, required=True)
parser.add_argument("--alpha_yn", type=float, default=0.3, help="Alpha value for regularization")
parser.add_argument("--lambda_reg", type=float, default=0.1, help="Regularization coefficient")
args = parser.parse_args()

fix_seed(args.seed)

device = 'cuda:0'

g, nclass, features, labels, train, val, test = preprocess_data(
    args.dataset, 
    args.train_ratio,
    args.remove_zero_in_degree_nodes,
    args.result_file_path,
    args.dataset_file_path
)

features = features.to(device)
labels = labels.to(device)
train = train.to(device)
test = test.to(device)
val = val.to(device)

g = g.to(device)
deg = g.in_degrees().float().clamp(min=1).to(device)
norm = torch.pow(deg, -0.5)
g.ndata['norm'] = norm.unsqueeze(1)  # Symmetric normalization

net = MixHopGCN(
    g=g,
    in_dim=features.size(1),
    hidden_dim=args.hidden,
    out_dim=nclass,
    num_hops=args.num_hops,
    combine=args.combine,
    dropout=args.dropout
).to(device)

# create optimizer
optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

# main loop
dur = []
los = []
loc = []
counter = 0
max_metric = 0.0

metric = accuracy if nclass > 2 else roc_auc

for epoch in range(args.epochs):
    if epoch >= 3:
        t0 = time.time()

    net.train()
    logp = net(features)

    cla_loss = F.nll_loss(logp[train], labels[train])

    loss = cla_loss
    train_metric = metric(logp[train], labels[train])

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    net.eval()
    logp = net(features)
    test_metric = metric(logp[test], labels[test])
    loss_val = F.nll_loss(logp[val], labels[val]).item()
    val_metric = metric(logp[val], labels[val])
    los.append([epoch, loss_val, val_metric, test_metric])

    if max_metric < val_metric:
        max_metric = val_metric
        counter = 0
    else:
        counter += 1

    if counter >= args.patience:
        print('early stop')
        break

    if epoch >= 3:
        dur.append(time.time() - t0)

if args.dataset in ['cora', 'citeseer', 'pubmed'] or 'syn' in args.dataset:
    los.sort(key=lambda x: x[1])
    acc = los[0][-1]
    print(f"Validation Loss-based Accuracy: {acc}")
else:
    los.sort(key=lambda x: -x[2])
    acc = los[0][-1]
    print(f"Validation Metric-based Accuracy: {acc}")

# Save model
model_filename = f"mixhop_model_hops{args.num_hops}_combine{args.combine}_depth{args.layer_num}_lr{args.lr}_wd{args.weight_decay}_seed{args.seed}.pth"
torch.save(net.state_dict(), os.path.join(args.result_file_path, model_filename))
print(f"Model saved: {model_filename}")