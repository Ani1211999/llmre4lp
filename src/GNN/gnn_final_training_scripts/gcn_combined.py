import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

from sklearn.metrics import roc_auc_score, average_precision_score
import argparse

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# GCN Encoder Model
class GCNEncoder(torch.nn.Module):
    def __init__(self, num_features, hidden_dim, embedding_dim):
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, embedding_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

# Link predictor
def predict_link(z, edge_label_index):
    src, dst = edge_label_index
    return (z[src] * z[dst]).sum(dim=1)

# Training function
def train(model, optimizer, data, pos_edges, neg_edges):
    model.train()
    optimizer.zero_grad()
    
    z = model(data)
    edge_label_index = torch.cat([pos_edges, neg_edges], dim=1)
    edge_label = torch.cat([
        torch.ones(pos_edges.shape[1]),
        torch.zeros(neg_edges.shape[1])
    ]).to(device)

    pred = predict_link(z, edge_label_index)
    loss = F.binary_cross_entropy_with_logits(pred, edge_label)
    loss.backward()
    optimizer.step()
    return loss.item()

# Evaluation function
@torch.no_grad()
def test(model, data, pos_edges, neg_edges):
    model.eval()
    z = model(data)
    edge_label_index = torch.cat([pos_edges, neg_edges], dim=1)
    edge_label = torch.cat([
        torch.ones(pos_edges.shape[1]),
        torch.zeros(neg_edges.shape[1])
    ]).to(device)

    pred = predict_link(z, edge_label_index).sigmoid()
    pred = pred.cpu().numpy()
    label = edge_label.cpu().numpy()

    auc = roc_auc_score(label, pred)
    ap = average_precision_score(label, pred)
    return auc, ap

# Main function
def main(npz_path):
    # Load processed data
    data_npz = np.load(npz_path, allow_pickle=True)
    node_features = torch.tensor(data_npz['node_features'], dtype=torch.float)
    edge_index = torch.tensor(data_npz['edges'], dtype=torch.long)

    # Load positive/negative edges
    train_edges = torch.tensor(data_npz['train_edges'], dtype=torch.long).to(device)
    val_edges = torch.tensor(data_npz['val_edges'], dtype=torch.long).to(device)
    test_edges = torch.tensor(data_npz['test_edges'], dtype=torch.long).to(device)

    train_neg_edges = torch.tensor(data_npz['train_neg_edges'], dtype=torch.long).to(device)
    val_neg_edges = torch.tensor(data_npz['val_neg_edges'], dtype=torch.long).to(device)
    test_neg_edges = torch.tensor(data_npz['test_neg_edges'], dtype=torch.long).to(device)

    # 3-hop edges
    hop3_test_edges = torch.tensor(data_npz['hop3_test_edges'], dtype=torch.long).to(device)
    hop3_test_neg_edges = torch.tensor(data_npz['hop3_test_neg_edges'], dtype=torch.long).to(device)

    # Build full graph (for reference)
    data_full = Data(x=node_features, edge_index=edge_index).to(device)

    # Build subgraph with only training edges (used for GNN propagation)
    train_edge_index_undirected = to_undirected(train_edges)
    data_train_only = Data(x=data_full.x, edge_index=train_edge_index_undirected).to(device)
    data_full = Data(x=node_features, edge_index=edge_index).to(device)
    print(data_train_only.is_undirected())  # Should return True
    # Initialize model
    model = GCNEncoder(
        num_features=data_train_only.num_features,
        hidden_dim=64,
        embedding_dim=64
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Training loop
    for epoch in range(1, 101):
        loss = train(model, optimizer, data_train_only, train_edges, train_neg_edges)
        val_auc, val_ap = test(model, data_train_only, val_edges, val_neg_edges)
        print(f"Epoch {epoch:03d}, Loss: {loss:.4f}, Val AUC: {val_auc:.4f}, Val AP: {val_ap:.4f}")

    # Test performance
    test_auc, test_ap = test(model, data_train_only, test_edges, test_neg_edges)
    print(f"\n1-Hop Test AUC: {test_auc:.4f}, AP: {test_ap:.4f}")

    # 3-hop evaluation
    hop3_auc, hop3_ap = test(model, data_train_only, hop3_test_edges, hop3_test_neg_edges)
    print(f"3-Hop Test AUC: {hop3_auc:.4f}, AP: {hop3_ap:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GNN for link prediction on Cora.")
    parser.add_argument('--npz_path', type=str, default='dataset/cora_processed.npz',
                        help="Path to preprocessed .npz file")
    args = parser.parse_args()
    main(args.npz_path)