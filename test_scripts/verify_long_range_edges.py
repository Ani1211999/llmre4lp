import json
import numpy as np
import networkx as nx
import os

def load_graph(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    edges = data['edges'].T  # Shape: (num_edges, 2)
    G = nx.Graph()
    G.add_edges_from(edges)
    return G, set(data['edges'][0]) | set(data['edges'][1])  # Set of all nodes in the graph

def verify_long_range_edges(npz_path, json_path):
    # Load the original graph and its nodes
    G, graph_nodes = load_graph(npz_path)

    # Load the long-range inference prompts
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Extract edges and labels
    edges = []
    labels = []
    missing_nodes = set()
    for item in data:
        u, v = map(int, item['id'].split('_'))
        label = item['conversations'][1]['value']
        edges.append((u, v))
        labels.append(label)
        if u not in graph_nodes or v not in graph_nodes:
            missing_nodes.add((u, v))

    if missing_nodes:
        print(f"Found {len(missing_nodes)} edges with missing nodes: {missing_nodes}")
        return

    # Verify path lengths
    path_lengths = []
    for u, v in edges:
        try:
            length = nx.shortest_path_length(G, u, v)
            path_lengths.append(length)
        except nx.NetworkXNoPath:
            path_lengths.append(float('inf'))  # Disconnected nodes
        except nx.NodeNotFound:
            path_lengths.append(float('inf'))  # Handle any remaining node issues

    # Analyze results
    print(f"Total edges in long_range_infer.json: {len(edges)}")
    print(f"Positive edges (labeled 'Yes'): {sum(1 for label in labels if label == 'Yes')}")
    print(f"Negative edges (labeled 'No'): {sum(1 for label in labels if label == 'No')}")
    print("\nPath Length Analysis:")
    print(f"Edges with path length >= 3: {sum(1 for length in path_lengths if length >= 3 or length == float('inf'))}")
    print(f"Edges with path length < 3: {sum(1 for length in path_lengths if length < 3 and length != float('inf'))}")
    print(f"Disconnected edges (path length = inf): {sum(1 for length in path_lengths if length == float('inf'))}")

    # Check for errors
    invalid_edges = [(u, v, length) for (u, v), length in zip(edges, path_lengths) if length < 3 and length != float('inf')]
    if invalid_edges:
        print("\nInvalid long-range edges (path length < 3):")
        for u, v, length in invalid_edges:
            print(f"Edge ({u}, {v}): Path length = {length}")

if __name__ == "__main__":
    npz_path = "./dataset/arxiv_2023.npz"
    json_path = "./llm_pred/prompt_json/Arxiv/long_range_infer.json"
    verify_long_range_edges(npz_path, json_path)