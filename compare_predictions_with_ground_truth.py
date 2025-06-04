import numpy as np
import json
import argparse
import os

def load_npz_data(npz_path):
    """Load data from the .npz file."""
    data = np.load(npz_path, allow_pickle=True)
    long_range_edges = data['long_range_edges']
    return long_range_edges

def load_prediction_data(prediction_path):
    """Load data from the prediction JSON file."""
    if not os.path.exists(prediction_path):
        print(f"Error: {prediction_path} not found.")
        return {}
    with open(prediction_path, 'r') as f:
        data = json.load(f)
    return {entry["id"]: float(entry["res"]) for entry in data}  # Convert res to float

def compare_predictions(npz_path, prediction_path, threshold=0.7, output_file=None):
    """Compare predicted probabilities with ground truth and filter edges with prob > threshold."""
    # Load data
    long_range_edges = load_npz_data(npz_path)
    predictions = load_prediction_data(prediction_path)

    # Convert ground truth to ID strings
    ground_truth_ids = [f"{u}_{v}" for u, v in long_range_edges.T]
    num_ground_truth = len(ground_truth_ids)
    print(f"Number of ground truth positive long-range edges: {num_ground_truth}")

    # Check coverage of ground truth in predictions
    covered_ids = [gid for gid in ground_truth_ids if gid in predictions]
    num_covered = len(covered_ids)
    print(f"Number of ground truth edges covered by predictions: {num_covered}")
    if num_covered < num_ground_truth:
        missing_ids = set(ground_truth_ids) - set(covered_ids)
        print(f"Number of missing ground truth edges: {len(missing_ids)}")
        print(f"Sample missing IDs: {list(missing_ids)[:5]}...")

    # Filter predictions with probability > threshold
    high_prob_edges = {id: prob for id, prob in predictions.items() if prob > threshold}
    num_high_prob = len(high_prob_edges)
    print(f"Number of edges with probability > {threshold}: {num_high_prob}")
    print(f"Sample high-probability edges: {list(high_prob_edges.keys())[:5]}...")

    # Identify true positives (high-prob edges that are in ground truth)
    true_positives = {id: prob for id, prob in high_prob_edges.items() if id in ground_truth_ids}
    num_true_positives = len(true_positives)
    print(f"Number of true positives (high-prob and in ground truth): {num_true_positives}")
    print(f"Sample true positives: {list(true_positives.keys())[:5]}...")

    # Calculate precision for high-prob edges
    precision = num_true_positives / num_high_prob if num_high_prob > 0 else 0.0
    print(f"Precision of high-probability edges: {precision:.4f}")

    # Save results if requested
    if output_file and high_prob_edges:
        with open(output_file, 'w') as f:
            json.dump({"threshold": threshold, "high_prob_edges": high_prob_edges, "true_positives": true_positives}, f, indent=2)
        print(f"Results saved to {output_file}")

    return high_prob_edges, true_positives

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz_path", type=str, default="dataset/arxiv_2023.npz", help="Path to the .npz file")
    parser.add_argument("--prediction_path", type=str, required=True, help="Path to the prediction JSON file")
    parser.add_argument("--threshold", type=float, default=0.7, help="Probability threshold for selecting edges")
    parser.add_argument("--output_file", type=str, default=None, help="Optional path to save the results as a JSON file")
    args = parser.parse_args()
    compare_predictions(args.npz_path, args.prediction_path, args.threshold, args.output_file)