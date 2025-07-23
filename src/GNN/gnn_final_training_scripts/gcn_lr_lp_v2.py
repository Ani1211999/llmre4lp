import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.utils.data import DataLoader, TensorDataset
import wandb

class GCNEncoder3Layer(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, out_channels)
        self.relu = torch.nn.ReLU()

    def forward(self, x, edge_index):
        x = self.relu(self.conv1(x, edge_index))
        x = self.relu(self.conv2(x, edge_index))
        x = self.conv3(x, edge_index)
        return x

def edge_score(embeddings, edge_index):
    src, dst = edge_index
    return (embeddings[src] * embeddings[dst]).sum(dim=1)

def evaluate_link_prediction(model, data, edges_pos, edges_neg, batch_size=1024, device='cpu'):
    model.eval()
    embeddings = model(data.x.to(device), data.edge_index.to(device))

    pos_scores = []
    neg_scores = []

    pos_loader = DataLoader(TensorDataset(edges_pos.t()), batch_size=batch_size, shuffle=False)
    neg_loader = DataLoader(TensorDataset(edges_neg.t()), batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for batch in pos_loader:
            batch_edges = batch[0].t().to(device)
            scores = edge_score(embeddings, batch_edges)
            pos_scores.append(scores.cpu())

        for batch in neg_loader:
            batch_edges = batch[0].t().to(device)
            scores = edge_score(embeddings, batch_edges)
            neg_scores.append(scores.cpu())

    pos_scores = torch.cat(pos_scores).numpy()
    neg_scores = torch.cat(neg_scores).numpy()

    labels = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
    scores = np.concatenate([pos_scores, neg_scores])

    auc = roc_auc_score(labels, scores)
    ap = average_precision_score(labels, scores)

    return auc, ap

def main():
    wandb.init(project="cora-3hop-link-prediction", config={
        "epochs": 50,
        "batch_size": 512,
        "lr": 0.01,
        "hidden_dim": 64,
        "embedding_dim": 32,
        "negative_ratio": 1.0,
        "model": "GCN 3-layer"
    })
    config = wandb.config

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    data_npz = np.load('../../dataset/cora_final_dataset.npz', allow_pickle=True)

    node_features = torch.tensor(data_npz['node_features'], dtype=torch.float)
    full_edge_index = torch.tensor(data_npz['edges'], dtype=torch.long)

    train_pos_edges = torch.tensor(data_npz['train_edges'], dtype=torch.long)
    train_neg_edges = torch.tensor(data_npz['train_neg_edges'], dtype=torch.long)

    hop3_test_pos_edges = torch.tensor(data_npz['hop3_test_edges'], dtype=torch.long)
    hop3_test_neg_edges = torch.tensor(data_npz['hop3_test_neg_edges'], dtype=torch.long)

    print(f"Train pos edges: {train_pos_edges.shape[1]}")
    print(f"Train neg edges: {train_neg_edges.shape[1]}")
    print(f"3-hop test pos edges: {hop3_test_pos_edges.shape[1]}")
    print(f"3-hop test neg edges: {hop3_test_neg_edges.shape[1]}")

    data = Data(x=node_features, edge_index=full_edge_index).to(device)

    model = GCNEncoder3Layer(in_channels=node_features.shape[1],
                             hidden_channels=config.hidden_dim,
                             out_channels=config.embedding_dim).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    train_edges = torch.cat([train_pos_edges, train_neg_edges], dim=1)
    train_labels = torch.cat([torch.ones(train_pos_edges.shape[1]), torch.zeros(train_neg_edges.shape[1])]).to(device)

    train_dataset = TensorDataset(train_edges.t(), train_labels)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

    best_auc = 0
    patience = 5
    epochs_no_improve = 0

    for epoch in range(1, config.epochs + 1):
        model.train()
        total_loss = 0

        for batch_edges, batch_labels in train_loader:
            batch_edges = batch_edges.t().to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            embeddings = model(data.x, data.edge_index)
            preds = edge_score(embeddings, batch_edges)
            loss = loss_fn(preds, batch_labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_edges.shape[1]

        avg_loss = total_loss / len(train_dataset)
        auc, ap = evaluate_link_prediction(model, data, hop3_test_pos_edges, hop3_test_neg_edges, batch_size=1024, device=device)

        print(f"Epoch {epoch:03d} | Train Loss: {avg_loss:.4f} | 3-hop Test AUC: {auc:.4f} | 3-hop Test AP: {ap:.4f}")

        # Log metrics to wandb
        wandb.log({
            "epoch": epoch,
            "train_loss": avg_loss,
            "3hop_test_auc": auc,
            "3hop_test_ap": ap
        })

        if auc > best_auc:
            best_auc = auc
            epochs_no_improve = 0
            torch.save(model.state_dict(), 'best_gcn_3layer_model.pth')
            wandb.run.summary["best_3hop_test_auc"] = best_auc
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping triggered.")
                break

    print(f"Training finished. Best 3-hop test AUC: {best_auc:.4f}")

    wandb.finish()

if __name__ == "__main__":
    main()
