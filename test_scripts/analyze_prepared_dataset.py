import numpy as np
import argparse

def analyze_dataset(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    
    # Extract arrays
    edges = data['edges']
    train_edges = data['train_edges']
    train_neg_edges = data['train_neg_edges']
    val_edges = data['val_edges']
    val_neg_edges = data['val_neg_edges']
    test_edges = data['test_edges']
    test_neg_edges = data['test_neg_edges']
    node_features = data['node_features']

    # Basic statistics
    num_nodes = np.max(edges) + 1 if edges.size > 0 else 2000
    print(f"Number of nodes: {num_nodes}")
    print(f"Original edges: {edges.shape[1]}")
    print(f"Train pos: {train_edges.shape[1]}, Train neg: {train_neg_edges.shape[1]}")
    print(f"Val pos: {val_edges.shape[1]}, Val neg: {val_neg_edges.shape[1]}")
    print(f"Test pos: {test_edges.shape[1]}, Test neg: {test_neg_edges.shape[1]}")
    print(f"Total pos edges: {train_edges.shape[1] + val_edges.shape[1] + test_edges.shape[1]}")
    print(f"Total neg edges: {train_neg_edges.shape[1] + val_neg_edges.shape[1] + test_neg_edges.shape[1]}")
    print(f"Positive:Negative ratio: {sum([train_edges.shape[1], val_edges.shape[1], test_edges.shape[1]]) / sum([train_neg_edges.shape[1], val_neg_edges.shape[1], test_neg_edges.shape[1]]):.2f}")
    print(f"Node features shape: {node_features.shape}")

    # Check for overlaps or duplicates
    all_pos_edges = np.concatenate([train_edges, val_edges, test_edges], axis=1)
    all_neg_edges = np.concatenate([train_neg_edges, val_neg_edges, test_neg_edges], axis=1)
    pos_set = set(map(tuple, all_pos_edges.T))
    neg_set = set(map(tuple, all_neg_edges.T))
    overlaps = pos_set & neg_set
    print(f"Number of positive-negative overlaps: {len(overlaps)}")
    duplicates = len(all_pos_edges.T) - len(set(map(tuple, all_pos_edges.T)))
    print(f"Number of duplicate positive edges: {duplicates}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--npz_path', required=True, help="Path to the .npz file")
    args = parser.parse_args()
    analyze_dataset(args.npz_path)