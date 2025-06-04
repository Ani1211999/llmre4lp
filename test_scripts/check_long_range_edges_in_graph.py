import numpy as np
import json
import argparse
import os

def load_npz_data(npz_path):
    """Load data from the .npz file."""
    data = np.load(npz_path, allow_pickle=True)
    edges = data['edges']
    return edges

def load_json_data(json_path):
    """Load data from the JSON file and extract edge IDs."""
    if not os.path.exists(json_path):
        print(f"Warning: {json_path} not found.")
        return []
    with open(json_path, 'r') as f:
        data = json.load(f)
    return [entry["id"] for entry in data]

def check_long_range_edges(npz_path, json_path):
    # Load original edges from .npz
    edges = load_npz_data(npz_path)
    original_edges_set = set(map(tuple, edges.T))  # Convert to undirected edges

    # Load long-range edges from JSON
    json_ids = load_json_data(json_path)
    if not json_ids:
        print("No data found in JSON file.")
        return

    # Assume the first 150 edges are positive long-range edges (1:5 ratio)
    positive_long_range_ids = json_ids[:150]
    total_edges = len(json_ids)
    print(f"Total edges in long_range_infer.json: {total_edges}")
    print(f"Number of positive long-range edges checked: {len(positive_long_range_ids)}")

    # Convert JSON IDs to (u, v) tuples
    positive_long_range_edges = [(int(id.split('_')[0]), int(id.split('_')[1])) for id in positive_long_range_ids]

    # Check for overlaps with original edges
    overlaps = [edge for edge in positive_long_range_edges if edge in original_edges_set or (edge[1], edge[0]) in original_edges_set]
    num_overlaps = len(overlaps)

    # Report results
    print(f"Number of positive long-range edges existing in original graph: {num_overlaps}")
    if overlaps:
        print(f"Sample overlapping edges: {overlaps[:5]}")
    else:
        print("No overlaps found.")

    if num_overlaps > 0:
        print("Warning: Some long-range edges already exist in the original graph. This may indicate an issue with edge generation or filtering.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz_path", type=str, default="dataset/arxiv_2023.npz", help="Path to the .npz file")
    parser.add_argument("--json_path", type=str, default="llm_pred/prompt_json/Arxiv/long_range_infer.json", help="Path to the long_range_infer.json file")
    args = parser.parse_args()
    check_long_range_edges(args.npz_path, args.json_path)