import numpy as np
import json
import os

# Load original graph data
data = np.load('./dataset/arxiv_2023.npz', allow_pickle=True)
edges = data['edges']
node_set = set(np.unique(edges))
original_edges_set = set(map(tuple, edges.T))

# Load rewired edges and probabilities
rewired_edges = np.load('src/GNN/llm_pred/rewired/rewired_edges.npy')
rewired_probs = np.load('src/GNN/llm_pred/rewired/rewired_probs.npy')

# Validate rewired edges
invalid_nodes = []
duplicates = set()
for i, (u, v) in enumerate(rewired_edges.T):
    if u not in node_set or v not in node_set:
        invalid_nodes.append((u, v))
    if (u, v) in duplicates or (v, u) in duplicates:
        duplicates.add((u, v))
    else:
        duplicates.add((u, v))

# Check consistency with predictions
with open('src/LLM/inference_results_lp_llm/long_range_edges/preds.json', 'r') as f:
    preds = json.load(f)
pred_ids = {pred['id']: pred['res'] for pred in preds}
mismatched_probs = []
for i, (u, v) in enumerate(rewired_edges.T):
    id_str = f"{u}_{v}"
    if id_str not in pred_ids or abs(rewired_probs[i] - pred_ids[id_str]) > 1e-6:
        mismatched_probs.append((id_str, rewired_probs[i], pred_ids.get(id_str)))

# Report results
print(f"Rewired edges shape: {rewired_edges.shape}, Probs shape: {rewired_probs.shape}")
print(f"Number of invalid nodes: {len(invalid_nodes)}")
if invalid_nodes:
    print(f"Sample invalid nodes: {invalid_nodes[:5]}")
print(f"Number of duplicate edges: {len(duplicates) - len(rewired_edges.T)}")
print(f"Number of mismatched probabilities: {len(mismatched_probs)}")
if mismatched_probs:
    print(f"Sample mismatches: {mismatched_probs[:5]}")