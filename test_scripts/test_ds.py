import json
import numpy as np

# Load original graph data
data = np.load('./dataset/arxiv_2023.npz', allow_pickle=True)
edges = data['edges']
node_set = set(np.unique(edges))

# Load LLM predictions
with open('llm_pred/prompt_json/Arxiv/long_range_infer.json', 'r') as f:
    predictions = json.load(f)

# Validate predicted edges
invalid_edges = []
for pred in predictions:
    try:
        u, v = map(int, pred['id'].split('_'))
        if u not in node_set or v not in node_set:
            invalid_edges.append(pred['id'])
        elif pred.get('score', 0) < 0 or pred.get('score', 0) > 1:
            invalid_edges.append(pred['id'])  # Invalid probability
    except ValueError:
        invalid_edges.append(pred['id'])  # Malformed ID

# Report results
print(f"Total predicted edges: {len(predictions)}")
print(f"Number of invalid edges: {len(invalid_edges)}")
if invalid_edges:
    print(f"Sample invalid edges: {invalid_edges[:5]}")
else:
    print("All predicted edges are valid.")

# Check for duplicates or overlaps with existing edges
original_edges_set = set(map(tuple, edges.T))
predicted_edges = [(int(pred['id'].split('_')[0]), int(pred['id'].split('_')[1])) for pred in predictions]
predicted_edges_set = set(predicted_edges)
overlaps = predicted_edges_set & original_edges_set
print(f"Number of overlaps with original edges: {len(overlaps)}")