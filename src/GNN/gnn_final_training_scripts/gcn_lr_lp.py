import os
import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
import dgl
from dgl.nn import GraphConv
import wandb

def fix_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class GCN(nn.Module):
    def __init__(self, in_feats, hidden_dim, out_dim, num_layers, dropout):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GraphConv(in_feats, hidden_dim, activation=F.relu))
        for _ in range(num_layers - 2):
            self.layers.append(GraphConv(hidden_dim, hidden_dim, activation=F.relu))
        self.layers.append(GraphConv(hidden_dim, out_dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, g, features):
        h = features
        for layer in self.layers[:-1]:
            h = self.dropout(h)
            h = layer(g, h)
        h = self.layers[-1](g, h)
        return h

class DotPredictor(nn.Module):
    def forward(self, h, edges):
        return (h[edges[:, 0]] * h[edges[:, 1]]).sum(dim=1)

def load_npz_data(npz_path):
    data = np.load(npz_path)
    return {
        'train_edges': torch.LongTensor(data['train_edges']),
        'train_neg_edges': torch.LongTensor(data['train_neg_edges']),
        'hop3_test_edges': torch.LongTensor(data['hop3_test_edges']),
        'hop3_test_neg_edges': torch.LongTensor(data['hop3_test_neg_edges']),
        'features': torch.FloatTensor(data['node_features'])
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--npz_path', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--wandb_project', type=str, default='GCN-LinkPred')
    parser.add_argument('--wandb_name', type=str, default=None)
    args = parser.parse_args()

    fix_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Initialize wandb
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_name,
        config=vars(args)
    )

    data = load_npz_data(args.npz_path)
    train_edges = data['train_edges']
    train_neg_edges = data['train_neg_edges']
    test_edges = data['hop3_test_edges']
    test_neg_edges = data['hop3_test_neg_edges']
    features = data['features'].to(device)

    num_nodes = features.shape[0]
    g = dgl.graph((train_edges[:, 0], train_edges[:, 1]), num_nodes=num_nodes)
    g = dgl.to_simple(g)
    g = dgl.to_bidirected(g)
    g = dgl.add_self_loop(g)
    g = g.to(device)

    train_edges = train_edges.to(device)
    train_neg_edges = train_neg_edges.to(device)
    test_edges = test_edges.to(device)
    test_neg_edges = test_neg_edges.to(device)
    model = GCN(
        in_feats=features.shape[1],
        hidden_dim=args.hidden,
        out_dim=args.hidden,  # for dot product
        num_layers=3,
        dropout=args.dropout
    ).to(device)

    predictor = DotPredictor().to(device)
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(predictor.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    best_auc = 0
    patience_counter = 0

    for epoch in range(args.epochs):
        model.train()
        h = model(g, features)

        pos_score = predictor(h, train_edges)
        neg_score = predictor(h, train_neg_edges)

        pred = torch.cat([pos_score, neg_score])
        true = torch.cat([
            torch.ones_like(pos_score),
            torch.zeros_like(neg_score)
        ])

        loss = F.binary_cross_entropy_with_logits(pred, true)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Eval on 3-hop test edges
        model.eval()
        with torch.no_grad():
            h = model(g, features)
            pos_score = predictor(h, test_edges)
            neg_score = predictor(h, test_neg_edges)

            preds = torch.cat([pos_score, neg_score])
            labels = torch.cat([
                torch.ones_like(pos_score),
                torch.zeros_like(neg_score)
            ])
            auc = roc_auc_score(labels.cpu(), preds.cpu())

        # Log to wandb
        wandb.log({
            'epoch': epoch,
            'train_loss': loss.item(),
            'test_auc': auc,
        })

        print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f} | Test AUC: {auc:.4f}")

        if auc > best_auc:
            best_auc = auc
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print("Early stopping.")
                break

    print(f"\nBest 3-hop Test AUC: {best_auc:.4f}")
    wandb.log({'best_test_auc': best_auc})
    wandb.finish()
