import numpy as np
import json
import argparse
import os

def load_npz_data(npz_path):
    """Load data from the .npz file."""
    data = np.load(npz_path, allow_pickle=True)
    edges = data['edges']
    train_edges = data['train_edges']
    val_edges = data['val_edges']
    test_edges = data['test_edges']
    train_neg_edges = data['train_neg_edges']
    val_neg_edges = data['val_neg_edges']
    test_neg_edges = data['test_neg_edges']
    long_range_edges = data['long_range_edges']
    long_range_neg_edges = data['long_range_neg_edges']
    node_texts = data['node_texts']
    return (edges, train_edges, val_edges, test_edges, train_neg_edges, val_neg_edges, test_neg_edges,
            long_range_edges, long_range_neg_edges, node_texts)

def load_json_data(json_path):
    """Load data from a JSON file and extract edge IDs."""
    if not os.path.exists(json_path):
        print(f"Warning: {json_path} not found.")
        return []
    with open(json_path, 'r') as f:
        data = json.load(f)
    return [entry["id"] for entry in data]

def verify_dataset(npz_path, save_dir):
    """Verify the consistency of JSON datasets against the .npz file."""
    # Load .npz data
    (edges, train_edges, val_edges, test_edges, train_neg_edges, val_neg_edges, test_neg_edges,
     long_range_edges, long_range_neg_edges, node_texts) = load_npz_data(npz_path)
    num_nodes = len(node_texts) if node_texts is not None else np.max(edges) + 1

    # Define expected JSON paths
    json_paths = {
        "train_all": os.path.join(save_dir, "train_all.json"),
        "val_all": os.path.join(save_dir, "val_all.json"),
        "test_all": os.path.join(save_dir, "test_all.json"),
        "long_range_infer": os.path.join(save_dir, "long_range_infer.json"),
    }

    # Convert .npz edge arrays to ID strings for comparison
    def edges_to_ids(edge_array):
        return [f"{u}_{v}" for u, v in edge_array.T]

    npz_edge_ids = {
        "train_all": edges_to_ids(train_edges) + edges_to_ids(train_neg_edges),
        "val_all": edges_to_ids(val_edges) + edges_to_ids(val_neg_edges),
        "test_all": edges_to_ids(test_edges) + edges_to_ids(test_neg_edges),
        "long_range_infer": edges_to_ids(long_range_edges) + edges_to_ids(long_range_neg_edges),
    }

    # Expected counts
    expected_counts = {
        "train_all": train_edges.shape[1] + train_neg_edges.shape[1],
        "val_all": val_edges.shape[1] + val_neg_edges.shape[1],
        "test_all": test_edges.shape[1] + test_neg_edges.shape[1],
        "long_range_infer": long_range_edges.shape[1] + long_range_neg_edges.shape[1],
    }

    # Verify each dataset
    for mode in json_paths.keys():
        json_ids = load_json_data(json_paths[mode])
        expected_ids = npz_edge_ids[mode]
        expected_count = expected_counts[mode]

        print(f"\nVerifying {mode}:")
        print(f"Expected number of edges: {expected_count}")
        print(f"Actual number of edges in JSON: {len(json_ids)}")

        # Check count match
        if len(json_ids) != expected_count:
            print(f"Warning: Mismatch in edge count! Expected {expected_count}, got {len(json_ids)}")
            missing = set(expected_ids) - set(json_ids)
            extra = set(json_ids) - set(expected_ids)
            if missing:
                print(f"Missing edges: {list(missing)[:5]}... (total {len(missing)})")
            if extra:
                print(f"Extra edges: {list(extra)[:5]}... (total {len(extra)})")
        else:
            print("Edge count matches.")

        # Check node ID ranges
        node_ids = set()
        for id_str in json_ids:
            u, v = map(int, id_str.split('_'))
            node_ids.add(u)
            node_ids.add(v)
        if max(node_ids) >= num_nodes or min(node_ids) < 0:
            print(f"Warning: Node IDs out of range [0, {num_nodes-1}]. Max ID: {max(node_ids)}")
        else:
            print("Node IDs are within valid range.")

        # Sample check for specific edges
        if json_ids and expected_ids:
            sample_idx = min(5, len(json_ids))
            for i in range(sample_idx):
                if json_ids[i] != expected_ids[i]:
                    print(f"Warning: Mismatch at index {i}: JSON '{json_ids[i]}' vs NPZ '{expected_ids[i]}'")
                else:
                    print(f"Match at index {i}: '{json_ids[i]}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz_path", type=str, default="dataset/arxiv_2023.npz", help="Path to the .npz file")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory containing JSON files")
    args = parser.parse_args()
    verify_dataset(args.npz_path, args.save_dir)