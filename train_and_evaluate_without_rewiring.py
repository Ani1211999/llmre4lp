import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import dgl
from dgl.nn import GatedGraphConv
import argparse
import os
from sklearn.metrics import roc_auc_score

class GatedGCN(nn.Module):
    def __init__(self, in_feats, hidden_size, num_layers, dropout):
        super(GatedGCN, self).__init__()
        self.input_projection = nn.Linear(in_feats, hidden_size)
        self.layers = nn.ModuleList()
        self.layers.append(GatedGraphConv(hidden_size, hidden_size, n_steps=1, n_etypes=1))
        for _ in range(num_layers - 1):
            self.layers.append(GatedGraphConv(hidden_size, hidden_size, n_steps=1, n_etypes=1))
        self.out_layer = nn.Linear(hidden_size * 2, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, g, features):
        h = self.input_projection(features)
        h = F.relu(h)
        for i, layer in enumerate(self.layers):
            h = layer(g, h)
            if i < len(self.layers) - 1:
                h = self.dropout(h)
        src, dst = g.edges()
        edge_embs = torch.cat([h[src], h[dst]], dim=-1)
        return torch.sigmoid(self.out_layer(edge_embs)).squeeze()

def train_model(model, g, features, train_edges, train_neg_edges, val_edges, val_neg_edges, test_edges, test_neg_edges, epochs, lr, patience):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_val_auc = 0
    patience_counter = 0
    log = []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        scores = model(g, features)

        train_pos_end = len(train_edges[0])
        train_neg_end = train_pos_end + len(train_neg_edges[0])
        train_pos_score = scores[:train_pos_end]
        train_neg_score = scores[train_pos_end:train_neg_end]

        pos_weight = torch.tensor([0.832 / 0.168]).to(scores.device)
        loss = F.binary_cross_entropy_with_logits(
            torch.cat([train_pos_score, train_neg_score]),
            torch.cat([torch.ones_like(train_pos_score), torch.zeros_like(train_neg_score)]),
            pos_weight=pos_weight
        )
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pos_end = train_neg_end + len(val_edges[0])
            val_neg_end = val_pos_end + len(val_neg_edges[0])
            val_pos_score = scores[train_neg_end:val_pos_end]
            val_neg_score = scores[val_pos_end:val_neg_end]
            val_labels = torch.cat([torch.ones_like(val_pos_score), torch.zeros_like(val_neg_score)])
            val_preds = torch.cat([val_pos_score, val_neg_score])
            val_auc = roc_auc_score(val_labels.cpu(), val_preds.cpu())
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                torch.save(model.state_dict(), "best_model.pth")
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        log.append(f"Epoch {epoch+1}/{epochs}, Val AUC: {val_auc:.4f}")

    model.load_state_dict(torch.load("best_model.pth"))
    model.eval()
    with torch.no_grad():
        scores = model(g, features)
        test_pos_end = val_neg_end + len(test_edges[0])
        test_neg_end = test_pos_end + len(test_neg_edges[0])
        test_pos_score = scores[val_neg_end:test_pos_end]
        test_neg_score = scores[test_pos_end:test_neg_end]
        test_labels = torch.cat([torch.ones_like(test_pos_score), torch.zeros_like(test_neg_score)])
        test_preds = torch.cat([test_pos_score, test_neg_score])
        test_auc = roc_auc_score(test_labels.cpu(), test_preds.cpu())
        log.append(f"Test AUC: {test_auc:.4f}")

    return log

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', required=True)
    parser.add_argument('--result_dir', required=True)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--hidden_size', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--patience', type=int, default=50)
    args = parser.parse_args()

    data = np.load(args.data_path, allow_pickle=True)
    edges = data['edges']
    train_edges = data['train_edges']
    train_neg_edges = data['train_neg_edges']
    val_edges = data['val_edges']
    val_neg_edges = data['val_neg_edges']
    test_edges = data['test_edges']
    test_neg_edges = data['test_neg_edges']
    features = torch.tensor(data['node_features'], dtype=torch.float32)

    print(f"Input feature dimension: {features.shape[1]}")
    print(f"Hidden size: {args.hidden_size}")

    all_edges = np.concatenate([
        edges,
        train_edges, val_edges, test_edges,
        train_neg_edges, val_neg_edges, test_neg_edges
    ], axis=1)
    g = dgl.graph((all_edges[0], all_edges[1]))
    g.ndata['feat'] = features

    model = GatedGCN(features.shape[1], args.hidden_size, args.num_layers, args.dropout)
    log = train_model(model, g, features, train_edges, train_neg_edges, val_edges, val_neg_edges, test_edges, test_neg_edges, args.epochs, args.lr, args.patience)

    os.makedirs(args.result_dir, exist_ok=True)
    with open(os.path.join(args.result_dir, "train_log.txt"), 'w') as f:
        f.write("\n".join(log))
    print(f"Model and logs saved to {args.result_dir}")

if __name__ == "__main__":
    main()
