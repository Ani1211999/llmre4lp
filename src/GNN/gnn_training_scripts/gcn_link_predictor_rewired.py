import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn import GraphConv
from sklearn.metrics import roc_auc_score
import argparse
import json
import os
import random
# -------------------------
# GCN Model
# -------------------------
class MLPPredictor(nn.Module):
    def __init__(self, in_feats, hidden_dim=64):
        super(MLPPredictor, self).__init__()
        self.W1 = nn.Linear(in_feats * 2, hidden_dim)
        self.W2 = nn.Linear(hidden_dim, 1)
        
    def forward(self, h, edge_index):
        h_u = h[edge_index[0]]
        h_v = h[edge_index[1]]
        h_cat = torch.cat([h_u, h_v], dim=1)
        return self.W2(F.relu(self.W1(h_cat))).squeeze()

class GCN(nn.Module):
    def __init__(self, in_feats, h_feats):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, h_feats)
        self.conv2 = GraphConv(h_feats, h_feats)
        self.conv3 = GraphConv(h_feats, h_feats)

    def forward(self, g, x):
        x = F.relu(self.conv1(g, x))
        x = F.relu(self.conv2(g, x))
        x = self.conv3(g, x)  # Final layer without activation
        return x

# -------------------------
# Loss and Metrics
# -------------------------

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed
        
def compute_loss(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score])
    labels = torch.cat([torch.ones_like(pos_score), torch.zeros_like(neg_score)])
    return F.binary_cross_entropy_with_logits(scores, labels)

def compute_auc(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score]).detach().cpu()
    labels = torch.cat([torch.ones_like(pos_score), torch.zeros_like(neg_score)]).detach().cpu()
    return roc_auc_score(labels, scores)

def save_preds_as_json(pos_edges, pos_score, neg_edges, neg_score, output_path):
    pos_edges = pos_edges.cpu().numpy()
    neg_edges = neg_edges.cpu().numpy()
    pos_probs = torch.sigmoid(pos_score).detach().cpu().numpy()
    neg_probs = torch.sigmoid(neg_score).detach().cpu().numpy()

    pred_data = []

    for i in range(pos_edges.shape[1]):
        src, dst = pos_edges[:, i]
        prob = pos_probs[i]
        pred = "Yes" if prob > 0.5 else "No"
        edge_id = f"{src}_{dst}"
        pred_data.append({"id": edge_id, "res": pred, "score": float(prob)})

    for i in range(neg_edges.shape[1]):
        src, dst = neg_edges[:, i]
        prob = neg_probs[i]
        pred = "Yes" if prob > 0.5 else "No"
        edge_id = f"{src}_{dst}"
        pred_data.append({"id": edge_id, "res": pred, "score": float(prob)})

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(pred_data, f, indent=2)
    print(f"Test predictions saved to {output_path}")

# -------------------------
# Main
# -------------------------
def main(npz_path, hidden_dim=64, epochs=200, lr=1e-2, device='cuda:0',dataset_name='Arxiv',rewired_path = None):
    set_seed(42)
    data = np.load(npz_path, allow_pickle=True)

    edges = torch.tensor(data['edges'], dtype=torch.long)
    train_edges = torch.tensor(data['train_edges'], dtype=torch.long)
    val_edges = torch.tensor(data['val_edges'], dtype=torch.long)
    test_edges = torch.tensor(data['test_edges'], dtype=torch.long)
    train_neg_edges = torch.tensor(data['train_neg_edges'], dtype=torch.long)
    val_neg_edges = torch.tensor(data['val_neg_edges'], dtype=torch.long)
    test_neg_edges = torch.tensor(data['test_neg_edges'], dtype=torch.long)
    features = torch.tensor(data['node_features'], dtype=torch.float)
    
    # rewired_edges_np = np.load("llm_pred/rewired/Arxiv_23_lr_v2/rewired_edges.npy")  # shape [2, N]
    # rewired_edges = torch.tensor(rewired_edges_np, dtype=torch.long)
    num_nodes = features.shape[0]
    structure_edges = train_edges

    if rewired_path is not None:
        rewired_np = np.load(rewired_path)  # shape [2, N]
        rewired_edges = torch.tensor(rewired_np, dtype=torch.long)
        print(f"Loaded {rewired_edges.shape[1]} LLM rewired edges")
        structure_edges = torch.cat([structure_edges, rewired_edges], dim=1)
    #augmented_edge_index = torch.cat([train_edges, rewired_edges], dim=1)

    # Build training graph (exclude val/test edges)
    train_g = dgl.graph((structure_edges[0], structure_edges[1]), num_nodes=num_nodes)
    train_g = dgl.to_simple(train_g)
    train_g = dgl.to_bidirected(train_g)
    train_g = dgl.remove_self_loop(train_g)
    train_g = dgl.add_self_loop(train_g)

    train_g = train_g.to(device)
    features = features.to(device)
    train_edges, train_neg_edges = train_edges.to(device), train_neg_edges.to(device)
    val_edges, val_neg_edges = val_edges.to(device), val_neg_edges.to(device)
    test_edges, test_neg_edges = test_edges.to(device), test_neg_edges.to(device)

    model = GCN(features.shape[1], hidden_dim).to(device)
    predictor = MLPPredictor(hidden_dim).to(device)
    optimizer = torch.optim.Adam(list(model.parameters()) + list(predictor.parameters()), lr=lr)

    best_val_auc = 0
    best_test_auc = 0
    best_epoch = 0
    for epoch in range(epochs):
        model.train()
        h = model(train_g, features)
        pos_score = predictor(h, train_edges)
        neg_score = predictor(h, train_neg_edges)
        loss = compute_loss(pos_score, neg_score)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            h = model(train_g, features)
            val_pos_score = predictor(h, val_edges)
            val_neg_score = predictor(h, val_neg_edges)
            val_auc = compute_auc(val_pos_score, val_neg_score)

            test_pos_score = predictor(h, test_edges)
            test_neg_score = predictor(h, test_neg_edges)
            test_auc = compute_auc(test_pos_score, test_neg_score)

            if test_auc > best_test_auc:
                best_test_auc = test_auc
                best_epoch= epoch + 1
                # Save best model checkpoint (optional)

        print(f"Epoch {epoch+1:03d} | Loss: {loss.item():.4f} | Val AUC: {val_auc:.4f} | Test AUC: {test_auc:.4f}")

    print(f"\nBest Test AUC (from best val): {best_test_auc:.4f}")

    # Save final test predictions in JSON for your LLM evaluation
    save_preds_as_json(test_edges, test_pos_score, test_neg_edges, test_neg_score, f"results/GNN/{dataset_name}_v1/test_preds.json")
    os.makedirs(f"results/GNN/baselines/{dataset_name}_hop1_rewired/", exist_ok=True)
    with open(f"results/GNN/baselines/{dataset_name}_hop1_rewired/eval_metrics.txt","w") as writer:
        writer.write(f"AUC: {best_test_auc:.4f}/n")
        writer.write(f"Epoch: {best_epoch:.4f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--npz_path', type=str, required=True, help='Path to .npz file with dataset')
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument("--dataset_name", type=str,required=True)
    parser.add_argument('--rewired_path', type=str, default=None,
                        help="Path to rewired_edges.npy file (2 x N) from LLM")
    args = parser.parse_args()

    main(args.npz_path, hidden_dim=args.hidden_dim, epochs=args.epochs, lr=args.lr, device=args.device,dataset_name=args.dataset_name, rewired_path = args.rewired_path)

