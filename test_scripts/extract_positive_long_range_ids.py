import numpy as np
import json
import argparse
import os

def load_npz_data(npz_path):
    """Load data from the .npz file."""
    data = np.load(npz_path, allow_pickle=True)
    long_range_edges = data['long_range_edges']
    return long_range_edges

def load_json_data(json_path):
    """Load data from the JSON file and extract edge IDs."""
    if not os.path.exists(json_path):
        print(f"Error: {json_path} not found.")
        return []
    with open(json_path, 'r') as f:
        data = json.load(f)
    return {entry["id"]: entry for entry in data}  # Dictionary for O(1) lookup

def extract_positive_long_range_ids(npz_path, json_path, output_file=None):
    """Extract edge IDs from long_range_infer.json that match positive long-range edges from .npz."""
    # Load data
    long_range_edges = load_npz_data(npz_path)
    json_data = load_json_data(json_path)

    # Convert long_range_edges to ID strings
    positive_ids = [f"{u}_{v}" for u, v in long_range_edges.T]
    num_positive_edges = len(positive_ids)
    print(f"Number of positive long-range edges in .npz: {num_positive_edges}")

    # Find matching IDs in JSON
    matched_ids = [pid for pid in positive_ids if pid in json_data]
    num_matched = len(matched_ids)
    print(f"Number of matched positive edge IDs in long_range_infer.json: {num_matched}")

    # Report mismatches or missing edges
    if num_matched < num_positive_edges:
        missing_ids = set(positive_ids) - set(matched_ids)
        print(f"Number of missing positive edges: {len(missing_ids)}")
        if missing_ids:
            print(f"Sample missing IDs: {list(missing_ids)[:5]}...")

    # Report sample matches
    print(f"Sample matched positive edge IDs: {matched_ids[:5]}..." if matched_ids else "No matches found.")

    # Save to file if requested
    if output_file and matched_ids:
        with open(output_file, 'w') as f:
            json.dump(matched_ids, f, indent=2)
        print(f"Positive edge IDs saved to {output_file}")

    return matched_ids

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz_path", type=str, default="dataset/arxiv_2023.npz", help="Path to the .npz file")
    parser.add_argument("--json_path", type=str, default="llm_pred/prompt_json/Arxiv/long_range_infer.json", help="Path to the long_range_infer.json file")
    parser.add_argument("--output_file", type=str, default=None, help="Optional path to save the positive edge IDs as a JSON file")
    args = parser.parse_args()
    extract_positive_long_range_ids(args.npz_path, args.json_path, args.output_file)