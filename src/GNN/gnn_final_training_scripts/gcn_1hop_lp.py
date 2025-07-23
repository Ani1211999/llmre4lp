import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_undirected
from sklearn.metrics import roc_auc_score, average_precision_score
import argparse
import wandb


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


def evaluate(model, x, edge_index, pos_edges, neg_edges, device):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--npz_path', type=str, required=True, help="Path to the .npz dataset")
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--out_dim', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--wandb_project', type=str, default='link-prediction-gnn')
    parser.add_argument('--wandb_name', type=str, default='run')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    wandb.init(project=args.wandb_project, name=args.wandb_name, config=vars(args))

    print("Loading dataset...")
    data = np.load(args.npz_path, allow_pickle=True)

    x = torch.tensor(data['node_features'], dtype=torch.float).to(device)
    train_edges = torch.tensor(data['train_edges'], dtype=torch.long).to(device)
    val_edges = torch.tensor(data['val_edges'], dtype=torch.long).to(device)
    test_edges = torch.tensor(data['test_edges'], dtype=torch.long).to(device)

    train_neg_edges = torch.tensor(data['train_neg_edges'], dtype=torch.long).to(device)
    val_neg_edges = torch.tensor(data['val_neg_edges'], dtype=torch.long).to(device)
    test_neg_edges = torch.tensor(data['test_neg_edges'], dtype=torch.long).to(device)

    edge_index = to_undirected(train_edges).to(device)  # message passing graph

    model = GCNEncoder(
        in_channels=x.size(1),
        hidden_channels=args.hidden_dim,
        out_channels=args.out_dim,
        dropout=args.dropout
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    patience = 20
    best_val_auc = 0
    best_test_auc = 0
    no_improvement = 0

    print("Training...")
    for epoch in range(1, args.epochs + 1):
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

        val_auc, val_ap = evaluate(model, x, edge_index, val_edges, val_neg_edges, device)
        test_auc, test_ap = evaluate(model, x, edge_index, test_edges, test_neg_edges, device)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_test_auc = test_auc
            no_improvement = 0
            
        else:
            no_improvement += 1
            if no_improvement >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
        wandb.log({
                'epoch': epoch,
                'loss': loss.item(),
                'val_auc': val_auc,
                'val_ap': val_ap,
                'test_auc': test_auc,
                'test_ap': test_ap,
                'best_val_auc': best_val_auc,
                'best_test_auc': best_test_auc
            })
        print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | Val AUC: {val_auc:.4f} | Test AUC: {test_auc:.4f}")

    print(f"✅ Best Val AUC: {best_val_auc:.4f}, Corresponding Test AUC: {best_test_auc:.4f}")


if __name__ == "__main__":
    main()
