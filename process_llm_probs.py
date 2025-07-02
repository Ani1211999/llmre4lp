import json
import numpy as np
import os
import argparse

def process_llm_probs(result_path, save_dir, threshold=0.5, max_edges=150):
    # Load the LLM inference results
    with open(result_path, 'r') as f:
        data_res = json.load(f)

    # Extract edges and probabilities
    edges = []
    probs = []
    for item in data_res:
        u, v = map(int, item['id'].split('_'))
        prob = float(item['res'])
        edges.append([u, v])
        probs.append(prob)

    # Sort edges by probability in descending order
    edges = np.array(edges)
    probs = np.array(probs)
    sorted_indices = np.argsort(probs)[::-1]
    edges = edges[sorted_indices]
    probs = probs[sorted_indices]

    # Select top edges with prob > threshold, up to max_edges
    selected_edges = []
    selected_probs = []
    for edge, prob in zip(edges, probs):
        if prob > threshold and len(selected_edges) < max_edges:
            selected_edges.append(edge)
            selected_probs.append(prob)
        if len(selected_edges) >= max_edges:
            break

    selected_edges = np.array(selected_edges).T  # Shape: (2, num_selected)
    selected_probs = np.array(selected_probs)

    # Save results
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    np.save(os.path.join(save_dir, 'rewired_edges.npy'), selected_edges)
    np.save(os.path.join(save_dir, 'rewired_probs.npy'), selected_probs)

    edge_prob_dict = {
        f"{u}_{v}": float(prob)
        for (u, v), prob in zip(selected_edges.T, selected_probs)
    }
    with open(os.path.join(save_dir, 'rewired_llm_probs.json'), 'w') as f:
        json.dump(edge_prob_dict, f, indent=2)
    print(f"Selected {len(selected_edges[0])} edges for rewiring with threshold {threshold}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_path", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max_edges", type=int, default=200)

    args = parser.parse_args()
    process_llm_probs(args.result_path, args.save_dir, args.threshold, args.max_edges)