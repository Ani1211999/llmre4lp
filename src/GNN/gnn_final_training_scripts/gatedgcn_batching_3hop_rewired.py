import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn import GatedGraphConv
from sklearn.metrics import roc_auc_score, average_precision_score
import wandb
import random

def set_seed(seed):
    """
    Sets the random seed for reproducibility across different libraries.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class GatedGCN(nn.Module):
    """
    Gated Graph Convolutional Network model for node embedding.

    Args:
        in_feats (int): Dimension of input node features.
        hidden_dim (int): Dimension of hidden layer embeddings.
        out_dim (int): Dimension of output node embeddings.
        n_steps (int): Number of message passing steps in GatedGraphConv.
        dropout (float): Dropout rate.
    """
    def __init__(self, in_feats, hidden_dim, out_dim, n_steps=3, dropout=0.5):
        super(GatedGCN, self).__init__()
        # Linear layer to project input features to hidden dimension
        self.linear_in = nn.Linear(in_feats, hidden_dim)
        # GatedGraphConv layer for message passing
        self.gnn = GatedGraphConv(
            in_feats=hidden_dim,
            out_feats=hidden_dim,
            n_steps=n_steps,
            n_etypes=1 # Assuming a single edge type for homogeneous graph
        )
        # Linear layer to project hidden embeddings to output dimension
        self.linear_out = nn.Linear(hidden_dim, out_dim)
        self.dropout = dropout

    def forward(self, g, x):
        """
        Forward pass of the GatedGCN model.

        Args:
            g (DGLGraph): The input graph.
            x (torch.Tensor): Input node features.

        Returns:
            torch.Tensor: Output node embeddings.
        """
        # Apply initial linear transformation
        x = self.linear_in(x)
        # Apply ReLU activation
        x = F.relu(x)
        # Apply dropout
        x = F.dropout(x, p=self.dropout, training=self.training)
        # Define edge types (all zeros for a single edge type)
        etype = torch.zeros(g.num_edges(), dtype=torch.long, device=g.device)
        # Apply GatedGraphConv
        x = self.gnn(g, x, etype)
        # Apply final linear transformation
        x = self.linear_out(x)
        return x

def dot_product_score(z, edge_index):
    """
    Calculates dot product scores for given edges based on node embeddings.

    Args:
        z (torch.Tensor): Node embeddings.
        edge_index (torch.Tensor): A 2xN tensor of edge indices (source, destination).

    Returns:
        torch.Tensor: Dot product scores for each edge.
    """
    # L2 normalize embeddings for better comparison
    z = F.normalize(z, p=2, dim=1)
    # Get embeddings for source nodes
    z_u = z[edge_index[0]]
    # Get embeddings for destination nodes
    z_v = z[edge_index[1]]
    # Compute dot product
    return (z_u * z_v).sum(dim=1)

@torch.no_grad()
def evaluate(model, g, x, pos_edges, neg_edges):
    """
    Evaluates the model's performance on link prediction.

    Args:
        model (nn.Module): The GNN model.
        g (DGLGraph): The input graph.
        x (torch.Tensor): Input node features.
        pos_edges (torch.Tensor): Positive edges (2xN).
        neg_edges (torch.Tensor): Negative edges (2xN).

    Returns:
        tuple: AUC and Average Precision scores.
    """
    model.eval() # Set model to evaluation mode
    z = model(g, x) # Compute full-graph embeddings
    # Calculate scores for positive and negative edges
    pos_scores = dot_product_score(z, pos_edges)
    neg_scores = dot_product_score(z, neg_edges)
    # Concatenate scores and labels for metric calculation
    scores = torch.cat([pos_scores, neg_scores], dim=0).cpu()
    labels = torch.cat([
        torch.ones(pos_scores.size(0)),
        torch.zeros(neg_scores.size(0))
    ], dim=0)
    # Calculate AUC and Average Precision
    auc = roc_auc_score(labels, scores)
    ap = average_precision_score(labels, scores)
    return auc, ap

class EdgeDataset(torch.utils.data.Dataset):
    """
    Dataset for batched edge training.

    Args:
        pos_edges (torch.Tensor): Positive edges (2xN).
        neg_edges (torch.Tensor): Negative edges (2xN).
    """
    def __init__(self, pos_edges, neg_edges):
        # Concatenate positive and negative edges
        self.edges = torch.cat([pos_edges, neg_edges], dim=1)
        # Create corresponding labels (1 for positive, 0 for negative)
        self.labels = torch.cat([
            torch.ones(pos_edges.size(1)),
            torch.zeros(neg_edges.size(1))
        ], dim=0)

    def __len__(self):
        """Returns the total number of edges in the dataset."""
        return self.edges.size(1)

    def __getitem__(self, idx):
        """
        Retrieves an edge and its label by index.

        Returns:
            tuple: (source_node_id, destination_node_id, label)
        """
        return self.edges[0, idx], self.edges[1, idx], self.labels[idx]

def run(config=None):
    """
    Main function to run the GatedGCN training and evaluation.

    Args:
        config (dict, optional): Configuration dictionary for wandb. Defaults to None.
    """
    with wandb.init(config=config):
        config = wandb.config
        set_seed(config.seed)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print("📂 Loading dataset...")
        # Load data from .npz file
        data = np.load(config.npz_path, allow_pickle=True)
        x = torch.tensor(data['node_features'], dtype=torch.float).to(device)
        train_edges = torch.tensor(data['train_edges'], dtype=torch.long).to(device)
        train_neg_edges = torch.tensor(data['train_neg_edges'], dtype=torch.long).to(device)
        test_3hop_edges = torch.tensor(data['hop3_test_edges'], dtype=torch.long).to(device)
        test_3hop_neg_edges = torch.tensor(data['hop3_test_neg_edges'], dtype=torch.long).to(device)

        # Load rewired edges and combine with training edges
        rewired_edges = torch.tensor(np.load(config.rewired_edges_path), dtype=torch.long).to(device)
        # Combine training edges and rewired edges, and add reverse edges for undirected graph
        edge_index = torch.cat([train_edges, rewired_edges], dim=1)
        edge_index = torch.cat([edge_index, edge_index[[1, 0]]], dim=1) # Add reverse edges

        num_nodes = x.size(0)
        # Create DGL graph
        g = dgl.graph((edge_index[0], edge_index[1]), num_nodes=num_nodes).to(device)

        # Initialize model
        model = GatedGCN(
            in_feats=x.size(1),
            hidden_dim=config.hidden_dim,
            out_dim=config.out_dim,
            dropout=config.dropout
        ).to(device)

        # Initialize optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

        # Create edge dataset and loader for batched edge training
        edge_dataset = EdgeDataset(train_edges, train_neg_edges)
        edge_loader = torch.utils.data.DataLoader(edge_dataset, batch_size=config.batch_size, shuffle=True)

        best_test_auc = 0
        best_epoch = 0
        no_improvement = 0
        patience = 20
        print("🚀 Starting batched training (Optimized)...")
        for epoch in range(1, config.epochs + 1):
            model.train() # Set model to training mode
            total_loss = 0

            for u, v, label in edge_loader:
                optimizer.zero_grad()

                # --- REVERTED CHANGE: Recompute full-graph embeddings for each batch ---
                # This is necessary for proper gradient flow when calling loss.backward()
                # for each batch, as the computational graph is consumed.
                z = model(g, x) # Recompute full-graph embeddings for each batch

                # Move batch data to device
                u = u.to(device)
                v = v.to(device)
                label = label.to(device)

                # Use the current batch's embeddings for score calculation
                scores = (z[u] * z[v]).sum(dim=1)
                loss = F.binary_cross_entropy_with_logits(scores, label)

                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(edge_loader)

            # Full-batch evaluation on 3-hop edges
            # The evaluate function already computes z internally, so no change needed here.
            test_auc, test_ap = evaluate(model, g, x, test_3hop_edges, test_3hop_neg_edges)

            # Early stopping logic
            if test_auc > best_test_auc:
                best_test_auc = test_auc
                best_epoch = epoch
                no_improvement = 0
            else:
                no_improvement += 1
                if no_improvement >= patience:
                    print(f"⏹️ Early stopping at epoch {epoch}")
                    break

            print(f"Epoch {epoch:03d} | Loss: {avg_loss:.4f} | 3-hop Test AUC: {test_auc:.4f}")
            # Log metrics to wandb
            wandb.log({
                'epoch': epoch,
                'loss': avg_loss,
                '3hop_test_auc': test_auc,
                '3hop_test_ap': test_ap,
                'best_3hop_test_auc': best_test_auc,
                'best_3hop_test_epoch': best_epoch,
                'seed': config.seed
            })


        print(f"\n✅ Best 3-hop Test AUC: {best_test_auc:.4f} at epoch {best_epoch}")

        # The result saving part is commented out as per your original code,
        # but you can uncomment and adjust if needed.
        # result_dir = f"results_rewired_sweep/{config.dataset_name}"
        # os.makedirs(result_dir, exist_ok=True)
        # result_file = os.path.join(result_dir, f"{config.dataset_name}_lr{config.lr}_hd{config.hidden_dim}_do{config.dropout}_seed{config.seed}.txt")
        # with open(result_file, "w") as f:
        #     f.write(f"Best 3-hop Test AUC: {best_test_auc:.4f}\n")
        #     f.write(f"Best Epoch: {best_epoch}\n")
        #     f.write(f"LR: {config.lr}, Dropout: {config.dropout}, Seed: {config.seed}\n")

if __name__ == "__main__":
    run({"lr": 0.0001,
    "hidden_dim": 512,
    "out_dim": 256,
    "dropout": 0.0,
    "seed": 42,
    "epochs": 1000,
    "batch_size": 2048,
    "npz_path": "../../dataset/Arxiv.npz",
    "rewired_edges_path": "./arxiv_rewired_edges/rewired_edges.npy",
    "dataset_name": "Arxiv",
    "wandb_project": "gatedgcn-lp-rewired",
    "wandb_name": "debug-fast"})
