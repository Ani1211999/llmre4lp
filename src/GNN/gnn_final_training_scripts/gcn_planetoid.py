import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv
from torch_geometric.utils import negative_sampling, train_test_split_edges
from sklearn.metrics import roc_auc_score
import numpy as np
# Load the Cora dataset and normalize features
dataset = Planetoid(root='./data/Cora', name='Cora', transform=T.NormalizeFeatures())
data = dataset[0]

# For link prediction, we'll split the edges.
# train_test_split_edges will add the following attributes to the data object:
# data.train_pos_edge_index: positive training edges
# data.val_pos_edge_index: positive validation edges
# data.val_neg_edge_index: negative validation edges (randomly sampled)
# data.test_pos_edge_index: positive test edges
# data.test_neg_edge_index: negative test edges (randomly sampled)
data.train_mask = data.val_mask = data.test_mask = None # Remove node classification masks
data = train_test_split_edges(data, val_ratio=0.1, test_ratio=0.1)

# Ensure data is on the correct device (GPU if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = data.to(device)

#     def __init__(self, in_channels, hidden_channels, out_channels):
#         super().__init__()
#         # First GCN layer: input features -> hidden features
#         self.conv1 = GCNConv(in_channels, hidden_channels)
#         # Second GCN layer: hidden features -> output embeddings
#         self.conv2 = GCNConv(hidden_channels, out_channels)

#     def encode(self, x, edge_index):
#         # Apply first GCN, then ReLU activation
#         x = self.conv1(x, edge_index).relu()
#         # Apply second GCN to get the final node embeddings
#         return self.conv2(x, edge_index)

#     def decode(self, z, pos_edge_index, neg_edge_index=None):
#         # Concatenate positive and negative edge indices for decoding
#         edge_index = pos_edge_index
#         if neg_edge_index is not None:
#             edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)

#         # Calculate inner product for all pairs of nodes specified by edge_index
#         # (z_u * z_v).sum() gives the dot product between embedding of node u and node v
#         return (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)

#     def decode_all(self, z):
#         # For evaluation, you might want to decode all possible pairs
#         # This creates a similarity matrix for all nodes
#         prob_adj = torch.sigmoid(z @ z.T) # Sigmoid to get probabilities
#         return prob_adj
class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        # First GCN layer: input features -> first hidden layer
        self.conv1 = GCNConv(in_channels, hidden_channels)
        # Second GCN layer: first hidden layer -> second hidden layer
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        # Third GCN layer: second hidden layer -> output embeddings
        self.conv3 = GCNConv(hidden_channels, out_channels)

    def encode(self, x, edge_index):
        # Apply first GCN with ReLU activation
        x = self.conv1(x, edge_index).relu()
        # Apply second GCN with ReLU activation
        x = self.conv2(x, edge_index).relu()
        # Apply third GCN to get final node embeddings
        return self.conv3(x, edge_index)

    def decode(self, z, pos_edge_index, neg_edge_index=None):
        # (Rest of the decode method remains the same)
        edge_index = pos_edge_index
        if neg_edge_index is not None:
            edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
        return (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)

    def decode_all(self, z):
        # (Rest of the decode_all method remains the same)
        prob_adj = torch.sigmoid(z @ z.T)
        return prob_adj
    # Initialize model, optimizer, and move to device

model = GCNEncoder(
    in_channels=dataset.num_features, # Input dimensions (node features)
    hidden_channels=128,              # Hidden layer dimensions
    out_channels=64                   # Output embedding dimensions
).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

def train():
    model.train()
    optimizer.zero_grad()

    # Get node embeddings using the training positive edges
    z = model.encode(data.x, data.train_pos_edge_index)

    # Sample negative edges for training
    # We sample the same number of negative edges as positive edges
    neg_edge_index = negative_sampling(
        edge_index=data.train_pos_edge_index,
        num_nodes=data.num_nodes,
        num_neg_samples=data.train_pos_edge_index.size(1),
        method='sparse', # 'sparse' method is generally efficient
    ).to(device)

    # Get scores for positive and negative edges
    pos_score = model.decode(z, data.train_pos_edge_index)
    neg_score = model.decode(z, neg_edge_index)

    # Combine scores and create corresponding labels (1 for positive, 0 for negative)
    scores = torch.cat([pos_score, neg_score], dim=0)
    labels = torch.cat([
        torch.ones(pos_score.size(0)),
        torch.zeros(neg_score.size(0))
    ]).to(device)

    # Calculate Binary Cross-Entropy Loss with Logits
    loss = F.binary_cross_entropy_with_logits(scores, labels)
    loss.backward()
    optimizer.step()
    return loss.item()

@torch.no_grad()
def test(pos_edge_index, neg_edge_index):
    model.eval()
    # Use the training graph for encoding to get node embeddings
    z = model.encode(data.x, data.train_pos_edge_index)

    # Get scores for positive and negative edges
    pos_score = model.decode(z, pos_edge_index)
    neg_score = model.decode(z, neg_edge_index)

    # Combine scores and labels for AUROC calculation
    scores = torch.cat([pos_score, neg_score], dim=0).cpu().numpy()
    labels = torch.cat([
        torch.ones(pos_score.size(0)),
        torch.zeros(neg_score.size(0))
    ]).cpu().numpy()

    # Calculate AUROC
    return roc_auc_score(labels, scores)

print("Starting GCN Link Prediction Training...")
for epoch in range(1, 201): # Train for 200 epochs
    loss = train()
    # Evaluate on validation set periodically
    
    val_auc = test(data.val_pos_edge_index, data.val_neg_edge_index)
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val_AUC: {val_auc:.4f}')

# After training, evaluate on the test set
test_auc = test(data.test_pos_edge_index, data.test_neg_edge_index)
print(f'Final Test_AUC: {test_auc:.4f}')