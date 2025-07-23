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
            n_etypes=1  # homogeneous graph
        )
        self.linear_out = nn.Linear(hidden_dim, out_dim)
        self.dropout = dropout

    def forward(self, g, x):
        x = self.linear_in(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        etype = torch.zeros(g.num_edges(), dtype=torch.long, device=g.device)  # homogeneous: all edges type 0
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


def run(config=None):
    with wandb.init(config=config):
        config = wandb.config
        set_seed(config.seed)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print("📂 Loading dataset...")
        data = np.load(config.npz_path, allow_pickle=True)

        x = torch.tensor(data['node_features'], dtype=torch.float).to(device)
        train_edges = torch.tensor(data['train_edges'], dtype=torch.long).to(device)
        val_edges = torch.tensor(data['val_edges'], dtype=torch.long).to(device)
        test_edges = torch.tensor(data['test_edges'], dtype=torch.long).to(device)

        train_neg_edges = torch.tensor(data['train_neg_edges'], dtype=torch.long).to(device)
        val_neg_edges = torch.tensor(data['val_neg_edges'], dtype=torch.long).to(device)
        test_neg_edges = torch.tensor(data['test_neg_edges'], dtype=torch.long).to(device)

        # 🔄 Add rewiring logic here
        if hasattr(config, 'rewired_edges_path') and config.rewired_edges_path:
            print("🔁 Loading rewired edges...")
            rewired_edges = torch.tensor(np.load(config.rewired_edges_path), dtype=torch.long).to(device)
            edge_index = torch.cat([train_edges, rewired_edges], dim=1)
        else:
            print("⚠️ No rewiring used.")
            edge_index = train_edges

        # Make edges undirected
        edge_index = torch.cat([edge_index, edge_index[[1, 0]]], dim=1)

        # Build DGL graph
        g = dgl.graph((edge_index[0], edge_index[1]), num_nodes=x.size(0)).to(device)

        model = GatedGCN(
            in_feats=x.shape[1],
            hidden_dim=config.hidden_dim,
            out_dim=config.out_dim,
            dropout=config.dropout
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

        best_val_auc = 0
        best_test_auc = 0
        best_epoch = 0
        no_improvement = 0
        patience = 20

        print("🚀 Starting training...")
        for epoch in range(1, config.epochs + 1):
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

            val_auc, val_ap = evaluate(model, g, x, val_edges, val_neg_edges)
            test_auc, test_ap = evaluate(model, g, x, test_edges, test_neg_edges)

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_test_auc = test_auc
                best_epoch = epoch
                no_improvement = 0
            else:
                no_improvement += 1
                if no_improvement >= patience:
                    print(f"⏹️ Early stopping at epoch {epoch}")
                    break

            print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | Val AUC: {val_auc:.4f} | Test AUC: {test_auc:.4f}")
            wandb.log({
                'epoch': epoch,
                'loss': loss.item(),
                'val_auc': val_auc,
                'val_ap': val_ap,
                'test_auc': test_auc,
                'test_ap': test_ap,
                'best_val_auc': best_val_auc,
                'best_test_auc': best_test_auc,
                'best_epoch': best_epoch,
                'seed': config.seed
            })

        print(f"\n✅ Best Val AUC: {best_val_auc:.4f}, Corresponding Test AUC: {best_test_auc:.4f}")
        # result_dir = f"results_rewired_1hop/{config.dataset_name}"
        # os.makedirs(result_dir, exist_ok=True)
        # result_file = os.path.join(
        #     result_dir,
        #     f"{config.dataset_name}_1hop_gatedgcn_lr{config.lr}_hd{config.hidden_dim}_do{config.dropout}_seed{config.seed}.txt"
        # )
        # with open(result_file, "w") as f:
        #     f.write(f"Best Val AUC: {best_val_auc:.4f}\n")
        #     f.write(f"Corresponding Test AUC: {best_test_auc:.4f}\n")
        #     f.write(f"Best Epoch: {best_epoch}\n")
        #     f.write(f"LR: {config.lr}, Dropout: {config.dropout}, Seed: {config.seed}\n")


if __name__ == "__main__":
    run()
