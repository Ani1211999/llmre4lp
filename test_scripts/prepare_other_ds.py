import pandas as pd
import torch
import numpy as np
import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected, remove_self_loops
import argparse
import os
import random

def load_data(dataset_name):
    from torch_geometric.datasets import Planetoid, CitationFull
    if dataset_name.lower() == "cora":
        dataset = Planetoid(root="../data", name="Cora")
    elif dataset_name.lower() == "pubmed":
        dataset = Planetoid(root="../data", name="PubMed")
    elif dataset_name.lower() == "dblp":
        dataset = CitationFull(root="../data", name="DBLPCitationNetwork")
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    return dataset[0]

def create_dummy_features(num_nodes, feature_dim=128):
    return torch.randn(num_nodes, feature_dim)

def sample_negative_edges(num_nodes, positive_edges, num_samples):
    positive_set = set(tuple(sorted([edge[0], edge[1]])) for edge in positive_edges.T)
    negative_edges = []
    while len(negative_edges) < num_samples:
        u = random.randint(0, num_nodes - 1)
        v = random.randint(0, num_nodes - 1)
        if u >= v:
            continue
        edge = tuple(sorted([u, v]))
        if edge not in positive_set:
            negative_edges.append([u, v])
    return np.array(negative_edges).T

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_npz_path', required=True)
    parser.add_argument('--dataset_name', required=True)
    parser.add_argument('--num_nodes_subgraph', type=int, default=2000)
    parser.add_argument('--negative_ratio', type=float, default=5.0)
    args = parser.parse_args()

    data = load_data(args.dataset_name)
    if data.is_directed():
        data.edge_index = to_undirected(data.edge_index)
    data.edge_index, _ = remove_self_loops(data.edge_index)

    # Subgraph extraction (simplified)
    if data.num_nodes > args.num_nodes_subgraph:
        indices = torch.randperm(data.num_nodes)[:args.num_nodes_subgraph]
        node_mapping = {old.item(): new for new, old in enumerate(indices)}
        edge_mask = torch.zeros(data.edge_index.shape[1], dtype=torch.bool)
        for i in range(data.edge_index.shape[1]):
            u, v = data.edge_index[:, i]
            if u in indices and v in indices:
                edge_mask[i] = 1
        new_edge_index = data.edge_index[:, edge_mask.bool]
        new_edge_index = torch.tensor([[node_mapping[edge.item()] for edge in new_edge_index[0]],
                                       [node_mapping[edge.item()] for edge in new_edge_index[1]]], dtype=torch.long)
        data = Data(
            edge_index=new_edge_index,
            num_nodes=args.num_nodes_subgraph,
            x=data.x[indices] if data.x is not None else create_dummy_features(args.num_nodes_subgraph)
        )

    edges = data.edge_index.cpu().numpy()
    num_edges = edges.shape[1]
    indices = torch.randperm(num_edges).numpy()
    train_end = int(num_edges * 0.8)
    val_end = int(num_edges * 0.9)
    train_edges = edges[:, indices[:train_end]]
    val_edges = edges[:, indices[train_end:val_end]]
    test_edges = edges[:, indices[val_end:]]

    num_train_neg = int(train_edges.shape[1] * args.negative_ratio)
    num_val_neg = int(val_edges.shape[1] * args.negative_ratio)
    num_test_neg = int(test_edges.shape[1] * args.negative_ratio)

    train_neg_edges = sample_negative_edges(data.num_nodes, edges, num_train_neg)
    val_neg_edges = sample_negative_edges(data.num_nodes, edges, num_val_neg)
    test_neg_edges = sample_negative_edges(data.num_nodes, edges, num_test_neg)

    node_texts = np.array([f"Dummy text for node {i}" for i in range(data.num_nodes)], dtype=object)
    node_labels = np.zeros(data.num_nodes, dtype=np.int32)

    np.savez(
        args.output_npz_path,
        edges=edges,
        train_edges=train_edges,
        val_edges=val_edges,
        test_edges=test_edges,
        train_neg_edges=train_neg_edges,
        val_neg_edges=val_neg_edges,
        test_neg_edges=test_neg_edges,
        node_features=data.x.cpu().numpy(),
        node_texts=node_texts,
        node_labels=node_labels
    )
    print(f"Saved dataset to {args.output_npz_path}")

if __name__ == "__main__":
    main()