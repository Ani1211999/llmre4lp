import json
import numpy as np
import networkx as nx
import os

def load_graph(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    edges = data['edges'].T  # Shape: (num_edges, 2)
    G = nx.Graph()
    G.add_edges_from(edges)
    # Set of all nodes in the graph (from edges)
    graph_nodes = set(data['edges'][0]) | set(data['edges'][1])
    # Set of all nodes in the subgraph (0 to num_nodes-1)
    all_nodes = set(range(data['node_texts'].shape[0]))
    # Isolated nodes
    isolated_nodes = all_nodes - graph_nodes
    # Set of edges for quick lookup
    edge_set = set(tuple(sorted([u, v])) for u, v in edges)
    return G, graph_nodes, all_nodes, isolated_nodes, edge_set, data

def verify_dataset(data, json_path, split_name, pos_key, neg_key, edge_set, all_nodes, isolated_nodes, check_path_length=False):
    # Load the prompts
    if not os.path.exists(json_path):
        print(f"{split_name} file not found: {json_path}")
        return

    with open(json_path, 'r') as f:
        prompts = json.load(f)

    # Extract edges and labels
    edges = []
    labels = []
    edges_with_missing_nodes = set()
    for item in prompts:
        u, v = map(int, item['id'].split('_'))
        label = item['conversations'][1]['value']
        edges.append((u, v))
        labels.append(label)
        if u not in all_nodes or v not in all_nodes:
            edges_with_missing_nodes.add((u, v))

    if edges_with_missing_nodes:
        print(f"\n{split_name} - Found {len(edges_with_missing_nodes)} edges with invalid nodes (outside 0 to {max(all_nodes)}): {edges_with_missing_nodes}")
        return

    # Verify positive/negative counts
    expected_pos = data[pos_key].shape[1]
    expected_neg = data[neg_key].shape[1]
    pos_count = sum(1 for label in labels if label == 'Yes')
    neg_count = sum(1 for label in labels if label == 'No')
    print(f"\n{split_name} - Total edges: {len(edges)}")
    print(f"{split_name} - Positive edges (labeled 'Yes'): {pos_count} (Expected: {expected_pos})")
    print(f"{split_name} - Negative edges (labeled 'No'): {neg_count} (Expected: {expected_neg})")

    if pos_count != expected_pos or neg_count != expected_neg:
        print(f"{split_name} - Mismatch in positive/negative counts!")

    # Check edge consistency with arxiv_2023.npz
    expected_pos_edges = set(tuple(sorted([u, v])) for u, v in data[pos_key].T)
    expected_neg_edges = set(tuple(sorted([u, v])) for u, v in data[neg_key].T)
    pos_edges = set(tuple(sorted([u, v])) for (u, v), label in zip(edges, labels) if label == 'Yes')
    neg_edges = set(tuple(sorted([u, v])) for (u, v), label in zip(edges, labels) if label == 'No')

    pos_mismatch = pos_edges - expected_pos_edges
    neg_mismatch = neg_edges - expected_neg_edges
    if pos_mismatch:
        print(f"{split_name} - Found {len(pos_mismatch)} positive edges not in {pos_key}: {pos_mismatch}")
    if neg_mismatch:
        print(f"{split_name} - Found {len(neg_mismatch)} negative edges not in {neg_key}: {neg_mismatch}")

    # Check isolated nodes in edges
    edges_with_isolated = [(u, v) for u, v in edges if u in isolated_nodes or v in isolated_nodes]
    if edges_with_isolated:
        print(f"{split_name} - Found {len(edges_with_isolated)} edges involving isolated nodes: {edges_with_isolated[:10]}...")

    # Check edge existence for positive/negative edges
    if not check_path_length:  # Train and Test splits
        pos_not_in_graph = [edge for edge in pos_edges if edge not in edge_set]
        neg_in_graph = [edge for edge in neg_edges if edge in edge_set]
        if pos_not_in_graph:
            print(f"{split_name} - Found {len(pos_not_in_graph)} positive edges not in the graph: {pos_not_in_graph[:10]}...")
        if neg_in_graph:
            print(f"{split_name} - Found {len(neg_in_graph)} negative edges that exist in the graph: {neg_in_graph[:10]}...")

    # Check path lengths for long-range edges (if applicable)
    if check_path_length:
        path_lengths = []
        for u, v in edges:
            try:
                length = nx.shortest_path_length(G, u, v)
                path_lengths.append(length)
            except nx.NetworkXNoPath:
                path_lengths.append(float('inf'))  # Disconnected nodes

        print(f"\n{split_name} - Path Length Analysis:")
        print(f"Edges with path length >= 3: {sum(1 for length in path_lengths if length >= 3 or length == float('inf'))}")
        print(f"Edges with path length < 3: {sum(1 for length in path_lengths if length < 3 and length != float('inf'))}")
        print(f"Disconnected edges (path length = inf): {sum(1 for length in path_lengths if length == float('inf'))}")

        # Check for invalid long-range edges
        invalid_edges = [(u, v, length) for (u, v), length in zip(edges, path_lengths) if length < 3 and length != float('inf')]
        if invalid_edges:
            print(f"\n{split_name} - Invalid long-range edges (path length < 3):")
            for u, v, length in invalid_edges:
                print(f"Edge ({u}, {v}): Path length = {length}")

if __name__ == "__main__":
    npz_path = "./dataset/arxiv_2023_v2.npz"
    train_json_path = "./llm_pred/prompt_json/Arxiv_v2/train_all.json"
    test_json_path = "./llm_pred/prompt_json/Arxiv_v2/test_all.json"
    long_range_json_path = "./llm_pred/prompt_json/Arxiv/long_range_infer.json"

    # Load graph and data
    G, graph_nodes, all_nodes, isolated_nodes, edge_set, data = load_graph(npz_path)
    print(f"Total nodes in subgraph: {len(all_nodes)}")
    print(f"Nodes with edges: {len(graph_nodes)}")
    print(f"Isolated nodes: {len(isolated_nodes)}")

    # Verify each dataset
    verify_dataset(data, train_json_path, "Train", "train_edges", "train_neg_edges", edge_set, all_nodes, isolated_nodes)
    verify_dataset(data, test_json_path, "Test", "test_edges", "test_neg_edges", edge_set, all_nodes, isolated_nodes)
    verify_dataset(data, long_range_json_path, "Long-Range", "long_range_edges", "long_range_neg_edges", edge_set, all_nodes, isolated_nodes, check_path_length=True)