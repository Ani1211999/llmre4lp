import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np
import argparse
import os

def make_undirected(edge_index):
    flipped = edge_index[[1, 0], :]
    edge_index = torch.cat([edge_index, flipped], dim=1)
    return edge_index

class GCNEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

class DotLinkPredictor(nn.Module):
    def forward(self, z, edge_index):
        src, dst = edge_index
        return (z[src] * z[dst]).sum(dim=-1)

def train(model, predictor, data, pos_edge_index, neg_edge_index, optimizer, device):
    model.train()
    predictor.train()
    optimizer.zero_grad()

    z = model(data.x.to(device), data.edge_index.to(device))
    pos_score = predictor(z, pos_edge_index.to(device))
    neg_score = predictor(z, neg_edge_index.to(device))

    pos_loss = -F.logsigmoid(pos_score).mean()
    neg_loss = -F.logsigmoid(-neg_score).mean()
    loss = pos_loss + neg_loss

    loss.backward()
    optimizer.step()
    return loss.item()

@torch.no_grad()
def evaluate(model, predictor, data, pos_edge_index, neg_edge_index, device):
    model.eval()
    predictor.eval()

    z = model(data.x.to(device), data.edge_index.to(device))
    pos_score = predictor(z, pos_edge_index.to(device)).sigmoid().cpu()
    neg_score = predictor(z, neg_edge_index.to(device)).sigmoid().cpu()

    scores = torch.cat([pos_score, neg_score])
    labels = torch.cat([torch.ones(pos_score.size(0)), torch.zeros(neg_score.size(0))])

    auc = roc_auc_score(labels, scores)
    ap = average_precision_score(labels, scores)
    return auc, ap

def edges_to_tensor(edge_array):
    return torch.tensor(edge_array, dtype=torch.long)

def build_pyg_graph(train_edges, node_features):
    edge_index = torch.tensor(train_edges, dtype=torch.long)
    edge_index = make_undirected(edge_index)
    x = torch.tensor(node_features, dtype=torch.float)
    return Data(x=x, edge_index=edge_index)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_rewired", action="store_true", help="Use LLM-rewired graph for training")
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--npz_path", type=str, required=True, help="Path to the preprocessed .npz file")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data_npz = np.load(args.npz_path, allow_pickle=True)
    node_features = data_npz["node_features"]

    if args.use_rewired:
        print("🧠 Using LLM-rewired graph")
        rewired_edges = np.concatenate([data_npz["train_edges"], data_npz["hop3_val_edges"]], axis=1)
        graph_data = build_pyg_graph(rewired_edges, node_features)
    else:
        print("📊 Using original graph")
        graph_data = build_pyg_graph(data_npz["train_edges"], node_features)

    train_pos = edges_to_tensor(data_npz["train_edges"])
    train_neg = edges_to_tensor(data_npz["train_neg_edges"])
    val_pos = edges_to_tensor(data_npz["val_edges"])
    val_neg = edges_to_tensor(data_npz["val_neg_edges"])
    test_pos = edges_to_tensor(data_npz["test_edges"])
    test_neg = edges_to_tensor(data_npz["test_neg_edges"])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GCNEncoder(in_channels=node_features.shape[1], hidden_channels=args.hidden_dim).to(device)
    predictor = DotLinkPredictor().to(device)
    optimizer = torch.optim.Adam(list(model.parameters()) + list(predictor.parameters()), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        loss = train(model, predictor, graph_data, train_pos, train_neg, optimizer, device)
        val_auc, val_ap = evaluate(model, predictor, graph_data, val_pos, val_neg, device)
        print(f"[Epoch {epoch:03d}] Loss: {loss:.4f} | Val AUC: {val_auc:.4f} | AP: {val_ap:.4f}")
    def to_undirected_edges(edge_tensor):
        u, v = edge_tensor
        edges = torch.cat([edge_tensor, torch.stack([v, u])], dim=1)
        return edges

    val_pos = to_undirected_edges(val_pos)
    val_neg = to_undirected_edges(val_neg)
    test_pos = to_undirected_edges(test_pos)
    test_neg = to_undirected_edges(test_neg)

    test_auc, test_ap = evaluate(model, predictor, graph_data, test_pos, test_neg, device)
    print(f"\n✅ Final Test AUC: {test_auc:.4f} | AP: {test_ap:.4f}")

if __name__ == "__main__":
    main()
