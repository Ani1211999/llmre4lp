import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_undirected
from sklearn.metrics import roc_auc_score, average_precision_score
import wandb
import random
import os


class GCNEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv3(x, edge_index)
        return x


def dot_product_score(z, edge_index):
    z = F.normalize(z, p=2, dim=1)
    z_u = z[edge_index[0]]
    z_v = z[edge_index[1]]
    return (z_u * z_v).sum(dim=1)


def evaluate(model, x, edge_index, pos_edges, neg_edges):
    model.eval()
    with torch.no_grad():
        z = model(x, edge_index)
        pos_scores = dot_product_score(z, pos_edges)
        neg_scores = dot_product_score(z, neg_edges)

        scores = torch.cat([pos_scores, neg_scores], dim=0).cpu()
        labels = torch.cat([
            torch.ones(pos_scores.size(0)),
            torch.zeros(neg_scores.size(0))
        ], dim=0)

        auc = roc_auc_score(labels, scores)
        ap = average_precision_score(labels, scores)
    return auc, ap


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run(config=None):
    with wandb.init(config=config):
        config = wandb.config
        set_seed(config.seed)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print("📦 Loading dataset...")
        data = np.load(config.npz_path, allow_pickle=True)

        x = torch.tensor(data['node_features'], dtype=torch.float).to(device)
        train_edges = torch.tensor(data['train_edges'], dtype=torch.long).to(device)
        train_neg_edges = torch.tensor(data['train_neg_edges'], dtype=torch.long).to(device)

        test_3hop_edges = torch.tensor(data['hop3_test_edges'], dtype=torch.long).to(device)
        test_3hop_neg_edges = torch.tensor(data['hop3_test_neg_edges'], dtype=torch.long).to(device)

        edge_index = to_undirected(train_edges).to(device)

        model = GCNEncoder(
            in_channels=x.size(1),
            hidden_channels=config.hidden_dim,
            out_channels=config.out_dim,
            dropout=config.dropout
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

        best_test_auc = 0
        best_epoch = 0
        no_improvement = 0
        patience = 20

        print("🚀 Training...")
        for epoch in range(1, config.epochs + 1):
            model.train()
            optimizer.zero_grad()

            z = model(x, edge_index)
            pos_score = dot_product_score(z, train_edges)
            neg_score = dot_product_score(z, train_neg_edges)

            scores = torch.cat([pos_score, neg_score], dim=0)
            labels = torch.cat([
                torch.ones(pos_score.size(0)),
                torch.zeros(neg_score.size(0))
            ], dim=0).to(device)

            loss = F.binary_cross_entropy_with_logits(scores, labels)
            loss.backward()
            optimizer.step()

            test_auc, test_ap = evaluate(model, x, edge_index, test_3hop_edges, test_3hop_neg_edges)

            if test_auc > best_test_auc:
                best_test_auc = test_auc
                best_epoch = epoch
                no_improvement = 0
            else:
                no_improvement += 1
                if no_improvement >= patience:
                    print(f"⏹️ Early stopping at epoch {epoch}")
                    break

            wandb.log({
                'epoch': epoch,
                'loss': loss.item(),
                '3hop_test_auc': test_auc,
                '3hop_test_ap': test_ap,
                'best_3hop_test_auc': best_test_auc,
                'best_3hop_test_epoch': best_epoch,
                'seed': config.seed
            })

            print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | 3-hop Test AUC: {test_auc:.4f}")

        print(f"\n✅ Best 3-hop Test AUC: {best_test_auc:.4f} at epoch {best_epoch}")

if __name__ == "__main__":
    run()
