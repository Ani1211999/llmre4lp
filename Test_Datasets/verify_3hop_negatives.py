import numpy as np
import torch
from torch_geometric.utils import to_undirected, coalesce, remove_self_loops, to_dense_adj
from collections import defaultdict

def edges_to_set(edges):
    return set(tuple(sorted(pair)) for pair in edges.T.tolist())

def get_exact_k_hop_edges(edge_index, num_nodes, k):
    """
    Returns a set of edges (u,v) that are connected via exactly k hops.
    """
    A = to_dense_adj(edge_index, max_num_nodes=num_nodes)[0].float()
    powers = {1: A}
    for i in range(2, k + 1):
        powers[i] = torch.mm(powers[i - 1], A)

    found_shorter = torch.zeros_like(A)
    k_path = (powers[k] > 0).float()
    for i in range(1, k):
        found_shorter += (powers[i] > 0).float()
    exact_k = (k_path - found_shorter) > 0
    edge_indices = torch.nonzero(exact_k, as_tuple=False)

    result_set = set()
    for u, v in edge_indices:
        if u.item() < v.item():
            result_set.add((u.item(), v.item()))
    return result_set

# === Load .npz file ===
npz_path = "../dataset/Arxiv.npz"  # Replace with your actual path
data = np.load(npz_path)

# === Get training graph (only train 1-hop edges) ===
train_edges = torch.tensor(data["train_edges"], dtype=torch.long)
train_edges = to_undirected(train_edges)
train_edges, _ = coalesce(train_edges, None, num_nodes=data["node_features"].shape[0])
train_edges, _ = remove_self_loops(train_edges)

# === Compute valid 3-hop positives from training graph ===
print("Computing exact 3-hop positives from training graph...")
valid_3hop_positives = get_exact_k_hop_edges(train_edges, num_nodes=data["node_features"].shape[0], k=3)

# === Load negatives to verify ===
hop3_train_neg = edges_to_set(data["hop3_train_neg_edges"])
hop3_test_neg = edges_to_set(data["hop3_test_neg_edges"])
hop3_val_neg = edges_to_set(data["hop3_val_neg_edges"])
# === Check for false negatives ===
overlap_train = hop3_train_neg.intersection(valid_3hop_positives)
overlap_test = hop3_test_neg.intersection(valid_3hop_positives)
overlap_val = hop3_val_neg.intersection(valid_3hop_positives)
print(f"\nValidation Results:")
print(f"❌ Invalid 3-hop train negatives (actually 3-hop positives): {len(overlap_train)}")
if overlap_train:
    print(f"Examples: {list(overlap_train)[:5]}")
print(f"❌ Invalid 3-hop test negatives (actually 3-hop positives): {len(overlap_test)}")
if overlap_test:
    print(f"Examples: {list(overlap_test)[:5]}")
print(f"❌ Invalid 3-hop val negatives (actually 3-hop positives): {len(overlap_val)}")
if overlap_val:
    print(f"Examples: {list(overlap_val)[:5]}")