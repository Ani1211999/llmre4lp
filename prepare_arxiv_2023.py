import pandas as pd
import torch
import numpy as np
import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected, coalesce, remove_self_loops
import argparse
import os
import sys
import random
from collections import defaultdict
import networkx as nx
import basics

# Set project path for LinkGPT utils (if needed)
project_path = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../.."))
if project_path not in sys.path:
    sys.path.insert(0, project_path)

# Load text data from CSV
def load_text_data(text_data_file_path):
    df = pd.read_csv(text_data_file_path)
    required_columns = ['node_id', 'title', 'abstract']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"CSV must contain columns: {required_columns}")
    return df

# Load graph data in PyG format
def load_graph_data(graph_data_file_path):
    if os.path.exists(graph_data_file_path):
        data = torch.load(graph_data_file_path)
        return data
    else:
        raise FileNotFoundError(f"Graph data file not found at {graph_data_file_path}")

# Return text and graph data
def load_tag_data(text_data_file_path, graph_data_file_path):
    graph = load_graph_data(graph_data_file_path)
    text = load_text_data(text_data_file_path)
    return graph, text

# Create dummy node features if none exist
def create_dummy_features(num_nodes, feature_dim=128):
    return torch.randn(num_nodes, feature_dim)

# Truncate text if it exceeds max_length
def truncate_text(text, max_length=1000):
    return text[:max_length] if len(text) > max_length else text

# Count self-loops in edge_index
def count_self_loops(edge_index):
    return (edge_index[0] == edge_index[1]).sum().item()

# Count duplicate edges in edge_index
def count_duplicates(edge_index):
    edges = edge_index.t().tolist()
    unique_edges = set(tuple(sorted(edge)) for edge in edges)  # Sort to treat (u,v) and (v,u) as same
    return len(edges) - len(unique_edges)

# Sample negative edges, excluding isolated nodes
def sample_negative_edges(num_nodes, positive_edges, num_samples, graph_nodes):
    positive_set = set(tuple(sorted([edge[0], edge[1]])) for edge in positive_edges.T)
    negative_edges = []
    attempts = 0
    max_attempts = num_samples * 10
    while len(negative_edges) < num_samples and attempts < max_attempts:
        u = random.randint(0, num_nodes - 1)
        v = random.randint(0, num_nodes - 1)
        if u >= v or u not in graph_nodes or v not in graph_nodes:
            continue
        edge = tuple(sorted([u, v]))
        if edge not in positive_set:
            negative_edges.append([u, v])
        attempts += 1
    return np.array(negative_edges).T if negative_edges else np.empty((2, 0), dtype=np.int64)

# Compute long-range edges (distance >= 3), excluding isolated nodes
def extract_long_range_edges(edges, num_nodes, graph_nodes, num_long_range=150):
    G = nx.Graph()
    G.add_edges_from(edges.T.tolist())
    
    long_range_edges = []
    for u in sorted(graph_nodes):
        for v in sorted(graph_nodes):
            if v <= u:
                continue
            try:
                dist = nx.shortest_path_length(G, u, v, method='dijkstra')
                if dist >= 3:
                    long_range_edges.append([u, v])
            except nx.NetworkXNoPath:
                continue  # Skip disconnected pairs within graph_nodes

    if len(long_range_edges) > num_long_range:
        long_range_edges = random.sample(long_range_edges, num_long_range)
    
    return np.array(long_range_edges).T if long_range_edges else np.empty((2, 0), dtype=np.int64)

# Extract a denser subgraph with improved connectivity
def extract_dense_subgraph(graph_data, df_text, node_id_field, num_nodes_subgraph=2000):
    # Step 1: Compute node degrees
    num_nodes = graph_data.num_nodes
    if num_nodes_subgraph >= num_nodes:
        print(f"Requested subgraph size ({num_nodes_subgraph}) is larger than or equal to original graph size ({num_nodes}). Using original graph.")
        return graph_data, df_text

    # Create a degree dictionary
    degrees = defaultdict(int)
    for i in range(graph_data.edge_index.shape[1]):
        u, v = graph_data.edge_index[0, i].item(), graph_data.edge_index[1, i].item()
        degrees[u] += 1
        degrees[v] += 1

    # Step 2: Select highest-degree nodes as initial set
    selected_nodes = sorted(degrees.keys(), key=lambda x: degrees[x], reverse=True)[:int(num_nodes_subgraph * 0.5)]
    visited = set(selected_nodes)

    # Step 3: Expand by adding neighbors iteratively, ensuring connectivity
    while len(visited) < num_nodes_subgraph and len(selected_nodes) < num_nodes:
        new_nodes = set()
        for node in selected_nodes:
            for i in range(graph_data.edge_index.shape[1]):
                neighbor = graph_data.edge_index[1, i].item() if graph_data.edge_index[0, i].item() == node else graph_data.edge_index[0, i].item() if graph_data.edge_index[1, i].item() == node else None
                if neighbor is not None and neighbor not in visited:
                    new_nodes.add(neighbor)
        if not new_nodes:
            break
        selected_nodes.extend(list(new_nodes - visited))
        visited.update(new_nodes)
        selected_nodes = selected_nodes[:num_nodes_subgraph]  # Limit to target size

    # If not enough connected nodes, add more based on degree (even if isolated)
    if len(visited) < num_nodes_subgraph:
        remaining = num_nodes_subgraph - len(visited)
        additional_nodes = [n for n in range(num_nodes) if n not in visited and n in degrees][:remaining]
        selected_nodes.extend(additional_nodes)
        visited.update(additional_nodes)

    selected_nodes = sorted(list(visited)[:num_nodes_subgraph])  # Ensure exact size

    # Step 4: Create a mapping from old node indices to new indices (0 to 1999)
    node_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(selected_nodes)}

    # Step 5: Filter edges to include only those between selected nodes
    edge_mask = torch.tensor([graph_data.edge_index[0, i].item() in selected_nodes and 
                              graph_data.edge_index[1, i].item() in selected_nodes 
                              for i in range(graph_data.edge_index.shape[1])], dtype=torch.bool)
    new_edge_index = graph_data.edge_index[:, edge_mask]

    # Step 6: Reindex the edges using the node mapping
    new_edge_index = torch.tensor([[node_mapping[edge.item()] for edge in new_edge_index[0]],
                                   [node_mapping[edge.item()] for edge in new_edge_index[1]]], dtype=torch.long)

    # Step 7: Update graph_data with the subgraph
    new_graph_data = Data(
        x=graph_data.x[selected_nodes] if graph_data.x is not None else None,
        edge_index=new_edge_index,
        num_nodes=num_nodes_subgraph
    )

    # Step 8: Filter df_text to include only the selected nodes
    df_text_subgraph = df_text[df_text[node_id_field].isin(selected_nodes)].copy()
    # Reindex node_id in df_text to match new indices
    df_text_subgraph[node_id_field] = df_text_subgraph[node_id_field].map(node_mapping)
    df_text_subgraph = df_text_subgraph.sort_values(node_id_field).reset_index(drop=True)

    return new_graph_data, df_text_subgraph, node_mapping

def main():
    basics.set_seeds(42)

    parser = argparse.ArgumentParser()
    parser.add_argument('--output_npz_path', required=True, help="Path to save the .npz file")
    parser.add_argument('--dataset_name', required=True)
    parser.add_argument('--text_data_file_path', required=True, help="Path to CSV file with text data")
    parser.add_argument('--graph_data_file_path', required=True, help="Path to .pt file with graph data")
    parser.add_argument('--node_id_field', default='node_id', help="Field name for node ID in CSV")
    parser.add_argument('--max_text_length', type=int, default=1000, help="Max length for combined text (default: 1000)")
    parser.add_argument('--num_nodes_subgraph', type=int, default=2000, help="Number of nodes in the subgraph (default: 2000)")
    parser.add_argument('--negative_ratio', type=float, default=5.0, help="Ratio of negative to positive edges (default: 5.0)")
    parser.add_argument('--num_long_range', type=int, default=150, help="Number of long-range edges for inference (default: 150)")
    
    args = parser.parse_args()

    DATASET_NAME = args.dataset_name
    text_data_file_path = args.text_data_file_path
    graph_data_file_path = args.graph_data_file_path
    node_id_field = args.node_id_field
    output_npz_path = args.output_npz_path
    max_text_length = args.max_text_length
    num_nodes_subgraph = args.num_nodes_subgraph
    negative_ratio = args.negative_ratio
    num_long_range = args.num_long_range

    # Load data
    graph_data, df_text = load_tag_data(text_data_file_path, graph_data_file_path)
    print("First 5 rows of text data (original):")
    print(df_text.head(5))

    # Preprocess graph data
    if graph_data.is_directed():
        graph_data.edge_index = to_undirected(graph_data.edge_index)
    print(f"After to_undirected: {graph_data.edge_index.shape[1]} edges")

    graph_data.edge_index, _ = coalesce(graph_data.edge_index, None, num_nodes=graph_data.num_nodes)
    print(f"After coalesce: {graph_data.edge_index.shape[1]} edges")

    graph_data.edge_index, _ = remove_self_loops(graph_data.edge_index)
    print(f"After remove_self_loops: {graph_data.edge_index.shape[1]} edges")

    print(f"Final self-loops: {count_self_loops(graph_data.edge_index)}")
    print(f"Final duplicates: {count_duplicates(graph_data.edge_index)}")

    # Extract denser subgraph
    print(f"Extracting denser subgraph with {num_nodes_subgraph} nodes...")
    graph_data, df_text, node_mapping = extract_dense_subgraph(graph_data, df_text, node_id_field, num_nodes_subgraph=num_nodes_subgraph)

    print(f"Number of nodes in subgraph: {graph_data.num_nodes}")
    print(f"Number of edges in subgraph: {graph_data.edge_index.shape[1]}")

    # Create node features if not present
    if not hasattr(graph_data, 'x') or graph_data.x is None:
        graph_data.x = create_dummy_features(graph_data.num_nodes, feature_dim=128)
    print(f"Shape of node features: {graph_data.x.shape}")

    # Prepare edges (2 x num_edges)
    edges = graph_data.edge_index.cpu().numpy()
    num_edges = edges.shape[1]
    graph_nodes = set(edges[0]) | set(edges[1])  # Nodes with edges

    # Split positive edges into train/val/test
    indices = torch.randperm(num_edges).numpy()
    train_end = int(num_edges * 0.8)
    val_end = int(num_edges * 0.9)

    train_edges = edges[:, indices[:train_end]]
    val_edges = edges[:, indices[train_end:val_end]]
    test_edges = edges[:, indices[val_end:]]

    print(f"Number of training edges: {train_edges.shape[1]}")
    print(f"Number of validation edges: {val_edges.shape[1]}")
    print(f"Number of test edges: {test_edges.shape[1]}")

    # Sample negative edges for each set
    num_train_neg = int(train_edges.shape[1] * negative_ratio)
    num_val_neg = int(val_edges.shape[1] * negative_ratio)
    num_test_neg = int(test_edges.shape[1] * negative_ratio)

    train_neg_edges = sample_negative_edges(graph_data.num_nodes, edges, num_train_neg, graph_nodes)
    val_neg_edges = sample_negative_edges(graph_data.num_nodes, edges, num_val_neg, graph_nodes)
    test_neg_edges = sample_negative_edges(graph_data.num_nodes, edges, num_test_neg, graph_nodes)

    print(f"Number of training negative edges: {train_neg_edges.shape[1]}")
    print(f"Number of validation negative edges: {val_neg_edges.shape[1]}")
    print(f"Number of test negative edges: {test_neg_edges.shape[1]}")

    # Compute long-range edges (distance >= 3)
    print(f"Computing {num_long_range} long-range edges (distance >= 3)...")
    long_range_edges = extract_long_range_edges(edges, graph_data.num_nodes, graph_nodes, num_long_range=num_long_range)
    num_long_range_neg = int(long_range_edges.shape[1] * negative_ratio)
    long_range_neg_edges = sample_negative_edges(graph_data.num_nodes, edges, num_long_range_neg, graph_nodes)

    print(f"Number of long-range edges: {long_range_edges.shape[1]}")
    print(f"Number of long-range negative edges: {long_range_neg_edges.shape[1]}")

    # Validate node_id range
    if df_text[node_id_field].max() >= graph_data.num_nodes:
        raise ValueError(f"node_id in CSV exceeds graph size: {df_text[node_id_field].max()} vs {graph_data.num_nodes}")

    # Prepare node_texts (combine title and abstract)
    df_text = df_text.sort_values(node_id_field)
    node_texts = [f"Title: {title} Abstract: {abstract}" for title, abstract in zip(df_text['title'], df_text['abstract'])]
    node_texts = [truncate_text(text, max_text_length) for text in node_texts]
    node_texts = np.array(node_texts, dtype=object)

    # Prepare node_labels (placeholder for link prediction)
    node_labels = np.zeros(graph_data.num_nodes, dtype=np.int32)  # Dummy labels since not used

    # Label texts (placeholder, minimal since not used)
    label_texts = np.array(['label_0'], dtype=object)

    # Save to .npz file
    np.savez(
        output_npz_path,
        edges=edges,  # Full edge set for reference
        train_edges=train_edges,
        val_edges=val_edges,
        test_edges=test_edges,
        train_neg_edges=train_neg_edges,
        val_neg_edges=val_neg_edges,
        test_neg_edges=test_neg_edges,
        long_range_edges=long_range_edges,
        long_range_neg_edges=long_range_neg_edges,
        node_labels=node_labels,
        node_features=graph_data.x.cpu().numpy(),
        node_texts=node_texts,
        label_texts=label_texts
    )
    print(f"Saved processed dataset with edge splits and long-range edges to {output_npz_path}")

if __name__ == '__main__':
    main()