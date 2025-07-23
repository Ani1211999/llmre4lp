import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import dgl
from dgl.nn import GatedGraphConv
from sklearn.metrics import roc_auc_score, average_precision_score
import argparse
import wandb
import random

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class GatedGCN(nn.Module):
    def __init__(self, in_feats, hidden_dim, out_dim, n_steps=3, dropout=0.5):
        super(GatedGCN, self).__init__()
        self.linear_in = nn.Linear(in_feats, hidden_dim)
        self.gnn = GatedGraphConv(
            in_feats=hidden_dim,
            out_feats=hidden_dim,
            n_steps=n_steps,
            n_etypes=1
        )
        self.linear_out = nn.Linear(hidden_dim, out_dim)
        self.dropout = dropout

    def forward(self, g, x):
        x = self.linear_in(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        etype = torch.zeros(g.num_edges(), dtype=torch.long, device=g.device)
        x = self.gnn(g, x, etype)
        x = self.linear_out(x)
        return x

def dot_product_score(z, edge_index):
    z = F.normalize(z, p=2, dim=1)
    z_u = z[edge_index[0]]
    z_v = z[edge_index[1]]
    return (z_u * z_v).sum(dim=1)

@torch.no_grad()
def evaluate(model, g, x, pos_edges, neg_edges):
    model.eval()
    z = model(g, x)
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

def compute_jacobian_norm(model, g, x, hop3_edges, sample_size=100):
    model.eval()
    x_jacobian = x.detach().clone()
    x_jacobian.requires_grad_(True)

    num_edges = hop3_edges.size(1)
    sample_size = min(sample_size, num_edges)
    perm = torch.randperm(num_edges)[:sample_size]

    z = model(g, x_jacobian)
    jacobian_norms = []

    for idx in perm:
        u, v = hop3_edges[:, idx]
        grad_output = torch.zeros_like(z)
        grad_output[u] = 1.0
        grad_input = autograd.grad(
            z,
            x_jacobian,
            grad_outputs=grad_output,
            create_graph=False,
            retain_graph=True
        )[0]

        if grad_input is not None:
            norm_val = torch.norm(grad_input[v]).item()
            jacobian_norms.append(norm_val)

    return sum(jacobian_norms) / len(jacobian_norms) if jacobian_norms else 0.0



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--npz_path', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--out_dim', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--wandb_project', type=str, default='link-prediction-3hop-gatedgcn')
    parser.add_argument('--wandb_name', type=str, default='gatedgcn-baseline-oversquashing')
    parser.add_argument('--dataset_name', type=str, required=True)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    wandb.init(config=vars(args))

    print("📂 Loading dataset...")
    data = np.load(args.npz_path, allow_pickle=True)

    x = torch.tensor(data['node_features'], dtype=torch.float).to(device)
    train_edges = torch.tensor(data['train_edges'], dtype=torch.long).to(device)
    train_neg_edges = torch.tensor(data['train_neg_edges'], dtype=torch.long).to(device)
    test_3hop_edges = torch.tensor(data['hop3_test_edges'], dtype=torch.long).to(device)
    test_3hop_neg_edges = torch.tensor(data['hop3_test_neg_edges'], dtype=torch.long).to(device)

    num_nodes = x.size(0)
    train_edges = torch.cat([train_edges, train_edges[[1, 0]]], dim=1)
    g = dgl.graph((train_edges[0], train_edges[1]), num_nodes=num_nodes).to(device)

    model = GatedGCN(
        in_feats=x.shape[1],
        hidden_dim=args.hidden_dim,
        out_dim=args.out_dim,
        dropout=args.dropout
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_test_auc = 0
    best_epoch = 0
    no_improvement = 0
    patience = 20

    print("🚀 Starting training...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()

        z = model(g, x)
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

        test_auc, test_ap = evaluate(model, g, x, test_3hop_edges, test_3hop_neg_edges)
        jacobian_norm = compute_jacobian_norm(model, g, x, test_3hop_edges)

        if test_auc > best_test_auc:
            best_test_auc = test_auc
            best_epoch = epoch
            no_improvement = 0
        else:
            no_improvement += 1
            if no_improvement >= patience:
                print(f"⏹️ Early stopping at epoch {epoch}")
                break

        print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | 3-hop Test AUC: {test_auc:.4f} | Jacobian norm: {jacobian_norm:.4f}")

        wandb.log({
            'epoch': epoch,
            'loss': loss.item(),
            '3hop_test_auc': test_auc,
            '3hop_test_ap': test_ap,
            'jacobian_norm': jacobian_norm,
            'best_3hop_test_auc': best_test_auc,
            'best_3hop_test_epoch': best_epoch
        })

    print(f"\n✅ Best 3-hop Test AUC: {best_test_auc:.4f} at epoch {best_epoch}")
    result_dir = "results_updated_scripts_final/"
    os.makedirs(result_dir, exist_ok=True)
    result_file = os.path.join(result_dir, f"{args.dataset_name}_lr{args.lr}_hd{args.hidden_dim}_do{args.dropout}_seed{args.seed}.txt")
    with open(result_file, "w") as f:
        f.write(f"Best 3-hop Test AUC: {best_test_auc:.4f}\n")
        f.write(f"Best Epoch: {best_epoch}\n")
        f.write(f"LR: {args.lr}, Dropout: {args.dropout}, Seed: {args.seed}\n")
        f.write(f"Final 3-hop Test AUC: {test_auc:.4f}\n")
        f.write(f"Final 3-hop Test AP: {test_ap:.4f}\n")
        f.write(f"Final Jacobian Norm: {jacobian_norm:.4f}\n")

if __name__ == "__main__":
    main()                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      