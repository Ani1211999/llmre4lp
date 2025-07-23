import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.metrics import roc_auc_score, average_precision_score
import wandb
from torch_geometric.utils import to_undirected

# 1. Define Evaluation Function (previously missing)
def evaluate_link_prediction(model, data, pos_edges, neg_edges, device='cpu'):
    """Evaluate model on positive/negative edges"""
    model.eval()
    with torch.no_grad():
        embeddings = model(data.x.to(device), data.edge_index.to(device))
        
        # Score positive edges
        pos_scores = (embeddings[pos_edges[0]] * embeddings[pos_edges[1]]).sum(dim=1).cpu().numpy()
        
        # Score negative edges
        neg_scores = (embeddings[neg_edges[0]] * embeddings[neg_edges[1]]).sum(dim=1).cpu().numpy()
    
    # Compute metrics
    y_true = np.concatenate([np.ones_like(pos_scores), np.zeros_like(neg_scores)])
    y_score = np.concatenate([pos_scores, neg_scores])
    
    auc = roc_auc_score(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    
    return auc, ap

# 2. GNN Model (Simplified for 1-hop)
class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.relu = torch.nn.ReLU()

    def forward(self, x, edge_index):
        x = self.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

# 3. Main Training Loop
def main():
    wandb.init(project="cora-1hop-link-prediction", config={
        "epochs": 200,
        "lr": 0.01,
        "hidden_dim": 128,
        "out_dim": 64,
        "patience": 10,
    })
    config = wandb.config
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load data (only 1-hop splits)
    data_npz = np.load('../../dataset/cora_final_dataset.npz', allow_pickle=True)
    train_edges = to_undirected(torch.tensor(data_npz['train_edges'], dtype=torch.long))
    val_edges = torch.tensor(data_npz['val_edges'], dtype=torch.long)
    test_edges = torch.tensor(data_npz['test_edges'], dtype=torch.long)
    train_neg_edges = torch.tensor(data_npz['train_neg_edges'], dtype=torch.long)
    val_neg_edges = torch.tensor(data_npz['val_neg_edges'], dtype=torch.long)
    test_neg_edges = torch.tensor(data_npz['test_neg_edges'], dtype=torch.long)
    node_features = torch.tensor(data_npz['node_features'], dtype=torch.float)

    # Create PyG Data object
    data = Data(x=node_features, edge_index=train_edges).to(device)

    # Model and optimizer
    model = GCNEncoder(
        in_channels=node_features.shape[1],
        hidden_channels=config.hidden_dim,
        out_channels=config.out_dim
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    # Prepare training data
    train_edge_pairs = torch.cat([train_edges, train_neg_edges], dim=1)
    train_labels = torch.cat([
        torch.ones(train_edges.shape[1]),
        torch.zeros(train_neg_edges.shape[1])
    ]).to(device)

    best_val_auc = 0.0
    epochs_no_improve = 0

    for epoch in range(1, config.epochs + 1):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        embeddings = model(data.x, data.edge_index)
        preds = (embeddings[train_edge_pairs[0]] * embeddings[train_edge_pairs[1]]).sum(dim=1)
        loss = loss_fn(preds, train_labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()

        # Validation
        val_auc, val_ap = evaluate_link_prediction(model, data, val_edges, val_neg_edges, device)
        wandb.log({
            "epoch": epoch,
            "train_loss": loss.item(),
            "val_auc": val_auc,
            "val_ap": val_ap
        })

        # Early stopping
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            epochs_no_improve = 0
            torch.save(model.state_dict(), 'best_1hop_model.pth')
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= config.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    # Final test evaluation
    model.load_state_dict(torch.load('best_1hop_model.pth'))
    test_auc, test_ap = evaluate_link_prediction(model, data, test_edges, test_neg_edges, device)
    print(f"\nFinal Test Performance:")
    print(f"AUC: {test_auc:.4f} | AP: {test_ap:.4f}")
    wandb.log({"test_auc": test_auc, "test_ap": test_ap})
    wandb.finish()

if __name__ == "__main__":
    main()