import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.utils.data import DataLoader, TensorDataset
import wandb
from torch_geometric.utils import to_undirected  # Critical for undirected graphs

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

    # Convert edge tensors to proper format
    edges_pos = edges_pos.to(device)
    edges_neg = edges_neg.to(device)

    with torch.no_grad():
        pos_scores = edge_score(embeddings, edges_pos).cpu().numpy()
        neg_scores = edge_score(embeddings, edges_neg).cpu().numpy()

    labels = np.concatenate([np.ones_like(pos_scores), np.zeros_like(neg_scores)])
    scores = np.concatenate([pos_scores, neg_scores])

    auc = roc_auc_score(labels, scores)
    ap = average_precision_score(labels, scores)

    return auc, ap

def main():
    wandb.init(project="cora-3hop-link-prediction", config={
        "epochs": 200,
        "batch_size": 512,
        "lr": 0.01,
        "hidden_dim": 128,  # Increased for better performance
        "embedding_dim": 64,
        "negative_ratio": 1.0,
        "model": "GCN-3Layer-Undirected",
        "patience": 10,
    })
    config = wandb.config
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load and prepare data
    data_npz = np.load('../../dataset/cora_final_dataset.npz', allow_pickle=True)
    
    # Convert to undirected graph explicitly
    train_edge_index = to_undirected(torch.tensor(data_npz['train_edges'], dtype=torch.long))
    node_features = torch.tensor(data_npz['node_features'], dtype=torch.float)
    
    # Prepare edge sets (already undirected in dataset)
    train_pos_edges = torch.tensor(data_npz['train_edges'], dtype=torch.long)
    train_neg_edges = torch.tensor(data_npz['train_neg_edges'], dtype=torch.long)
    hop3_test_pos_edges = torch.tensor(data_npz['hop3_test_edges'], dtype=torch.long)
    hop3_test_neg_edges = torch.tensor(data_npz['hop3_test_neg_edges'], dtype=torch.long)

    # Create PyG Data object
    data = Data(x=node_features, edge_index=train_edge_index).to(device)

    # Verify edge directions
    print(f"Training graph edges: {data.edge_index.shape[1]} (should be ~2x {train_pos_edges.shape[1]})")
    unique_edges = set()
    for u, v in data.edge_index.t().tolist():
        unique_edges.add((min(u,v), max(u,v)))
    print(f"Unique edges in training graph: {len(unique_edges)}")

    # Model and optimizer
    model = GCNEncoder3Layer(
        in_channels=node_features.shape[1],
        hidden_channels=config.hidden_dim,
        out_channels=config.embedding_dim
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=5e-4)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    # Prepare training data
    train_edges = torch.cat([train_pos_edges, train_neg_edges], dim=1)
    train_labels = torch.cat([
        torch.ones(train_pos_edges.shape[1]),
        torch.zeros(train_neg_edges.shape[1])
    ])
    
    # Shuffle training data
    shuffle_idx = torch.randperm(train_edges.size(1))
    train_edges = train_edges[:, shuffle_idx]
    train_labels = train_labels[shuffle_idx].to(device)

    best_auc = 0.0
    epochs_no_improve = 0

    for epoch in range(1, config.epochs + 1):
        model.train()
        total_loss = 0.0
        
        # Batch training
        for i in range(0, train_edges.size(1), config.batch_size):
            batch_edges = train_edges[:, i:i+config.batch_size].to(device)
            batch_labels = train_labels[i:i+config.batch_size]
            
            optimizer.zero_grad()
            embeddings = model(data.x, data.edge_index)
            preds = edge_score(embeddings, batch_edges)
            loss = loss_fn(preds, batch_labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * batch_edges.shape[1]

        avg_loss = total_loss / train_edges.size(1)
        
        # Evaluation
        auc, ap = evaluate_link_prediction(
            model, data, 
            hop3_test_pos_edges, hop3_test_neg_edges,
            device=device
        )
        
        print(f"Epoch {epoch:03d} | Loss: {avg_loss:.4f} | 3-hop Test AUC: {auc:.4f} | AP: {ap:.4f}")
        wandb.log({
            "epoch": epoch,
            "train_loss": avg_loss,
            "3hop_test_auc": auc,
            "3hop_test_ap": ap
        })

        # Early stopping
        if auc > best_auc:
            best_auc = auc
            epochs_no_improve = 0
            torch.save(model.state_dict(), 'best_gcn_model.pth')
            wandb.run.summary["best_3hop_test_auc"] = best_auc
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= config.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    print(f"Training complete. Best 3-hop test AUC: {best_auc:.4f}")
    wandb.finish()

if __name__ == "__main__":
    main()