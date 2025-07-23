import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.utils import negative_sampling
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np
import random
import argparse
import os

# --- GCN Model Definition (3-Layer) ---
class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels_1, hidden_channels_2, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels_1)
        self.conv2 = GCNConv(hidden_channels_1, hidden_channels_2)
        self.conv3 = GCNConv(hidden_channels_2, out_channels) # Added third layer

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        return self.conv3(x, edge_index)

    def decode(self, z, edge_index):
        return (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)

    def decode_all(self, z):
        prob_adj = torch.sigmoid(z @ z.T)
        return prob_adj

# --- GNN Training and Evaluation Functions ---

def train_gnn(model, data, optimizer, num_epochs, device):
    """
    Trains the GNN model for link prediction.
    """
    print("\nStarting GNN Training...")
    for epoch in range(1, num_epochs + 1):
        model.train()
        optimizer.zero_grad()

        z = model.encode(data.x, data.train_pos_edge_index)

        # Sample negative edges for training
        neg_edge_index = negative_sampling(
            edge_index=data.train_pos_edge_index,
            num_nodes=data.num_nodes,
            num_neg_samples=data.train_pos_edge_index.size(1),
            method='sparse',
        ).to(device)

        pos_scores = model.decode(z, data.train_pos_edge_index)
        neg_scores = model.decode(z, neg_edge_index)

        scores = torch.cat([pos_scores, neg_scores], dim=0)
        labels = torch.cat([
            torch.ones(pos_scores.size(0)),
            torch.zeros(neg_scores.size(0))
        ]).to(device)

        loss = F.binary_cross_entropy_with_logits(scores, labels)
        
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0:
            val_auc, val_ap = evaluate_gnn(model, data.x, data.train_pos_edge_index, data.val_pos_edge_index, data.val_neg_edge_index, data.num_nodes, device)
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val_AUC: {val_auc:.4f}, Val_AP: {val_ap:.4f}')
    print("GNN Training Complete. 🎉")

#----------------------------------------------------------------------------------------------------

@torch.no_grad()
def evaluate_gnn(model, x, encoder_edge_index, pos_edge_index, neg_edge_index, num_nodes, device):
    """
    Evaluates the GNN model's performance on a given set of edges.
    """
    model.eval()

    z = model.encode(x, encoder_edge_index)

    pos_score = model.decode(z, pos_edge_index)
    neg_score = model.decode(z, neg_edge_index)

    scores = torch.cat([pos_score, neg_score], dim=0).cpu().numpy()
    labels = torch.cat([
        torch.ones(pos_score.size(0)),
        torch.zeros(neg_score.size(0))
    ]).cpu().numpy()

    auc_score = roc_auc_score(labels, scores)
    ap_score = average_precision_score(labels, scores)
    return auc_score, ap_score

#----------------------------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="GNN Link Prediction from preprocessed NPZ file.")
    parser.add_argument('--npz_path', type=str, default='dataset/cora_processed.npz',
                        help="Path to the preprocessed data NPZ file.")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument('--epochs', type=int, default=200, help="Number of training epochs for GNN.")

    args = parser.parse_args()
    
    # Set seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- Load Data from NPZ File ---
    print(f"Loading preprocessed data from: {args.npz_path}")
    if not os.path.exists(args.npz_path):
        raise FileNotFoundError(f"Error: NPZ file not found at {args.npz_path}. Please ensure it's generated first.")
    
    data_dict = np.load(args.npz_path, allow_pickle=True)
    # Convert numpy arrays to torch tensors for GPU compatibility if needed
    data_dict = {k: torch.tensor(v, dtype=torch.long) if 'edge' in k else torch.tensor(v, dtype=torch.float)
                 for k, v in data_dict.items()}
    # Convert node_features to float as it contains continuous features
    data_dict['node_features'] = data_dict['node_features'].float()
    
    print("Data loaded successfully.")
    print(f"Node features shape: {data_dict['node_features'].shape}")
    print(f"Train 1-hop edges shape: {data_dict['train_edges'].shape}")
    print(f"Val 1-hop edges shape: {data_dict['val_edges'].shape}")
    print(f"Test 1-hop edges shape: {data_dict['test_edges'].shape}")
    print(f"Val 1-hop neg edges shape: {data_dict['val_neg_edges'].shape}")
    print(f"Test 1-hop neg edges shape: {data_dict['test_neg_edges'].shape}")
    
    num_nodes = data_dict['node_features'].shape[0]

    # Prepare Data object for GNN training using the 1-hop splits
    gnn_data = Data(x=data_dict['node_features'], num_nodes=num_nodes)
    gnn_data.train_pos_edge_index = data_dict['train_edges'].to(device)
    gnn_data.val_pos_edge_index = data_dict['val_edges'].to(device)
    gnn_data.val_neg_edge_index = data_dict['val_neg_edges'].to(device)
    gnn_data.test_pos_edge_index = data_dict['test_edges'].to(device)
    gnn_data.test_neg_edge_index = data_dict['test_neg_edges'].to(device)

    # --- GNN Training and Evaluation (1-hop only) ---
    print("\n--- Starting GNN for 1-hop Link Prediction ---")
    model = GCNEncoder(
        in_channels=gnn_data.x.shape[1],
        hidden_channels_1=128,
        hidden_channels_2=64,
        out_channels=32 # Dimensionality of node embeddings
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Train the GNN on 1-hop training edges
    # The `train_pos_edge_index` is used both for message passing and as positive labels for training
    train_gnn(model, gnn_data, optimizer, num_epochs=args.epochs, device=device)

    # Evaluate the trained GNN on 1-hop test edges
    print("\n--- Final Evaluation on 1-hop Test Set ---")
    # GNN encodes based on the graph it was trained on (train_pos_edge_index)
    final_test_auc, final_test_ap = evaluate_gnn(
        model, gnn_data.x, gnn_data.train_pos_edge_index,
        gnn_data.test_pos_edge_index, gnn_data.test_neg_edge_index,
        gnn_data.num_nodes, device
    )
    print(f"1-hop Test AUC: {final_test_auc:.4f}, Test AP: {final_test_ap:.4f}")

if __name__ == "__main__":
    main()