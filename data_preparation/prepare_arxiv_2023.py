import pandas as pd
import torch
import numpy as np
import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj, to_undirected, coalesce, remove_self_loops
import argparse
import os
import random
from collections import defaultdict, deque
import networkx as nx
import basics

# Assuming 'basics.py' contains set_seeds

def edges_to_set(edges):
    """Convert edge array to set of sorted (u, v) pairs"""
    if edges.size == 0:
        return set()
    return set(tuple(sorted(pair)) for pair in edges.T.tolist())

def extract_dense_subgraph(graph_data, df_text, node_id_field, num_nodes_subgraph=2000):
    """
    Extract a connected, degree-balanced subgraph
    Args:
        graph_data: Original PyG graph data
        df_text: DataFrame with node text information
        node_id_field: Column name for node IDs in df_text
        num_nodes_subgraph: Target number of nodes
    Returns:
        (subgraph, filtered_text_df, node_mapping)
    """
    num_nodes = graph_data.num_nodes
    if num_nodes_subgraph >= num_nodes:
        print(f"Requested subgraph size ({num_nodes_subgraph}) is larger than or equal to original graph size ({num_nodes}). Using original graph.")
        # Ensure mapping is identity if full graph is used
        node_mapping = {i: i for i in range(num_nodes)}
        return graph_data, df_text.copy(), node_mapping

    # Create a degree dictionary
    degrees = defaultdict(int)
    # Ensure edge_index is a tensor before iterating
    if not isinstance(graph_data.edge_index, torch.Tensor):
        graph_data.edge_index = torch.tensor(graph_data.edge_index, dtype=torch.long)
    
    # Calculate degrees for the undirected graph
    temp_edge_index = to_undirected(graph_data.edge_index)
    for i in range(temp_edge_index.shape[1]):
        u, v = temp_edge_index[0, i].item(), temp_edge_index[1, i].item()
        degrees[u] += 1

    # Step 2: Select highest-degree nodes as initial set
    # Sort by degree, then by node ID for deterministic behavior
    sorted_nodes_by_degree = sorted(degrees.keys(), key=lambda x: (degrees[x], x), reverse=True)
    
    # Start with a percentage of highest-degree nodes
    initial_selection_size = int(num_nodes_subgraph * 0.1) # Start with 10% highest degree
    selected_nodes = sorted_nodes_by_degree[:initial_selection_size]
    visited = set(selected_nodes)
    
    # Create a NetworkX graph from the PyG data for BFS
    nx_graph = nx.Graph()
    nx_graph.add_nodes_from(range(num_nodes)) # Add all nodes first
    nx_graph.add_edges_from(graph_data.edge_index.t().tolist())

    # Step 3: Expand by adding neighbors iteratively (BFS-like)
    q = deque(selected_nodes)
    
    while len(visited) < num_nodes_subgraph and q:
        current_node = q.popleft()
        
        for neighbor in nx_graph.neighbors(current_node):
            if neighbor not in visited:
                if len(visited) < num_nodes_subgraph: # Only add if space available
                    visited.add(neighbor)
                    q.append(neighbor)
                else:
                    break # Stop adding if target size reached
        if len(visited) >= num_nodes_subgraph:
            break

    # If we still don't have enough nodes, add more high-degree nodes (not necessarily connected to current set)
    if len(visited) < num_nodes_subgraph:
        remaining_needed = num_nodes_subgraph - len(visited)
        additional_nodes = [n for n in sorted_nodes_by_degree if n not in visited][:remaining_needed]
        visited.update(additional_nodes)
        print(f"Added {len(additional_nodes)} disconnected high-degree nodes to reach target size.")

    selected_nodes_final = sorted(list(visited)) # Ensure sorted for deterministic mapping

    # Step 4: Create a mapping from old node indices to new indices (0 to num_nodes_subgraph-1)
    node_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(selected_nodes_final)}

    # Step 5: Filter edges to include only those between selected nodes
    new_edges = []
    for u, v in graph_data.edge_index.t().tolist():
        if u in selected_nodes_final and v in selected_nodes_final:
            new_edges.append([node_mapping[u], node_mapping[v]])
    new_edge_index = torch.tensor(new_edges, dtype=torch.long).t()

    # Step 6: Update graph_data with the subgraph
    subgraph_x = None
    if hasattr(graph_data, 'x') and graph_data.x is not None:
        subgraph_x = graph_data.x[selected_nodes_final]

    new_graph_data = Data(
        x=subgraph_x,
        edge_index=new_edge_index,
        num_nodes=len(selected_nodes_final) # Use actual count of selected nodes
    )
    # Coalesce and remove self loops for the new subgraph
    new_graph_data.edge_index = to_undirected(new_graph_data.edge_index)
    new_graph_data.edge_index, _ = coalesce(new_graph_data.edge_index, None, num_nodes=new_graph_data.num_nodes)
    new_graph_data.edge_index, _ = remove_self_loops(new_graph_data.edge_index)
    print(f"Subgraph created with {new_graph_data.num_nodes} nodes and {new_graph_data.edge_index.shape[1]} edges.")


    # Step 7: Filter df_text to include only the selected nodes and reindex
    df_text_subgraph = df_text[df_text[node_id_field].isin(selected_nodes_final)].copy()
    df_text_subgraph['original_node_id'] = df_text_subgraph[node_id_field] 
    df_text_subgraph[node_id_field] = df_text_subgraph[node_id_field].map(node_mapping)
    df_text_subgraph = df_text_subgraph.sort_values(node_id_field).reset_index(drop=True)

    return new_graph_data, df_text_subgraph, node_mapping

def split_disjoint_edges(all_edges_numpy, train_ratio=0.8, val_ratio=0.1):
    """
    Splits edges ensuring no overlap between train, val, test.
    Handles (u,v) and (v,u) as the same edge.
    Args:
        all_edges_numpy: numpy array of shape (2, E)
    Returns:
        train_edges, val_edges, test_edges (numpy arrays, 2xM)
    """
    if all_edges_numpy.size == 0:
        return np.empty((2, 0), dtype=int), np.empty((2, 0), dtype=int), np.empty((2, 0), dtype=int)

    # Convert to set of sorted tuples to handle undirected nature
    edge_set = set(tuple(sorted(pair)) for pair in all_edges_numpy.T)
    edge_list = list(edge_set)
    random.shuffle(edge_list)

    n_total = len(edge_list)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_edges = np.array(edge_list[:n_train]).T if n_train > 0 else np.empty((2,0), dtype=int)
    val_edges = np.array(edge_list[n_train:n_train + n_val]).T if n_val > 0 else np.empty((2,0), dtype=int)
    test_edges = np.array(edge_list[n_train + n_val:]).T if (n_total - n_train - n_val) > 0 else np.empty((2,0), dtype=int)
    
    # Ensure no duplicates if input was not unique (should be handled by set conversion, but good safeguard)
    train_edges = np.unique(train_edges, axis=1) if train_edges.size > 0 else np.empty((2, 0), dtype=int)
    val_edges = np.unique(val_edges, axis=1) if val_edges.size > 0 else np.empty((2, 0), dtype=int)
    test_edges = np.unique(test_edges, axis=1) if test_edges.size > 0 else np.empty((2, 0), dtype=int)

    return train_edges, val_edges, test_edges


def sample_negative_edges(num_nodes, positive_edges_set, num_samples, graph_nodes_in_subgraph, max_attempts_multiplier=100):
    """
    Sample non-existent edges as negative examples.
    Ensures sampled negatives are not in positive_edges_set and are within graph_nodes_in_subgraph.
    Args:
        num_nodes: Total number of nodes in the *subgraph*.
        positive_edges_set: A set of existing edges (u,v) where u<v.
        num_samples: How many negatives to generate.
        graph_nodes_in_subgraph: List of node IDs present in the current subgraph.
        max_attempts_multiplier: Multiplier for max attempts to find negatives.
    Returns:
        Negative edges (2 x num_samples) as a numpy array.
    """
    if len(graph_nodes_in_subgraph) < 2:
        return np.empty((2, 0), dtype=int)

    negatives = []
    attempts = 0
    max_attempts = num_samples * max_attempts_multiplier
    
    graph_nodes_list = list(graph_nodes_in_subgraph)

    while len(negatives) < num_samples and attempts < max_attempts:
        u, v = random.sample(graph_nodes_list, 2)
        if u == v: # No self-loops
            attempts += 1
            continue
        
        # Ensure (u,v) is sorted for consistent lookup
        u, v = (u, v) if u < v else (v, u)

        if (u, v) not in positive_edges_set:
            negatives.append([u, v])
        attempts += 1
    
    return np.array(negatives).T if negatives else np.empty((2, 0), dtype=int)


def load_text_data(text_data_file_path):
    """Load node text data from CSV with validation"""
    df = pd.read_csv(text_data_file_path)
    required_columns = ['node_id', 'title', 'abstract']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"CSV must contain columns: {required_columns}")
    return df

def load_graph_data(graph_data_file_path):
    """Load PyG graph data with existence check"""
    if os.path.exists(graph_data_file_path):
        return torch.load(graph_data_file_path)
    raise FileNotFoundError(f"Graph data not found at {graph_data_file_path}")

def load_tag_data(text_data_file_path, graph_data_file_path):
    graph = load_graph_data(graph_data_file_path)
    text = load_text_data(text_data_file_path)
    return graph, text

def create_dummy_features(num_nodes, feature_dim=128):
    """Create random features if none exist"""
    return torch.randn(num_nodes, feature_dim)


def get_k_hop_edges_matrix_power(edge_index, num_nodes, max_k, exclude_edges_set=None):
    """
    Extracts k-hop connections (paths of length k) using matrix powers.
    Returns:
        A dictionary where keys are hop distances (k) and values are
        numpy arrays of (2, N) shape, representing k-hop edges.
        (u, v) means u and v are connected by a path of length k,
        and no shorter path, and (u,v) is not in exclude_edges_set.
    """
    if edge_index.numel() == 0: # Handle empty edge_index
        return defaultdict(lambda: np.empty((2,0), dtype=int))

    adj_matrix = to_dense_adj(edge_index, max_num_nodes=num_nodes)[0].float()
    
    # Zero out excluded edges in the adjacency matrix
    if exclude_edges_set:
        temp_adj = adj_matrix.clone()
        for u, v in exclude_edges_set:
            # Check if within bounds before setting to 0
            if u < num_nodes and v < num_nodes: 
                temp_adj[u, v] = 0
                temp_adj[v, u] = 0 # Undirected
        adj_matrix = temp_adj

    A_powers = {1: adj_matrix}
    # Compute A^k
    for k in range(2, max_k + 1):
        A_powers[k] = torch.mm(A_powers[k-1], adj_matrix)

    final_k_hop_edges_sets = defaultdict(set)
    
    # This matrix keeps track of nodes that have *already* been connected by a shorter path.
    # It effectively stores the union of A^1, A^2, ..., A^(k-1) after each step.
    shortest_path_found_matrix = torch.zeros_like(adj_matrix) 
    
    for k_val in range(1, max_k + 1):
        # Current connectivity at exactly k hops, before considering shorter paths
        current_k_connectivity = (A_powers[k_val] > 0).float()
        
        # Paths of exact length k must exist in A^k AND not exist in any previous shortest path matrix
        # (i.e., not connected by a shorter path)
        exact_k_path_mask = (current_k_connectivity - shortest_path_found_matrix) > 0
        
        # Update shortest_path_found_matrix for the next iteration: add current k-hop paths
        shortest_path_found_matrix = shortest_path_found_matrix + current_k_connectivity
        shortest_path_found_matrix = (shortest_path_found_matrix > 0).float() # Keep it binary (0 or 1)

        # Extract pairs for the current exact hop distance
        # We store all k-hop paths here, then sample from them later
        pairs = torch.nonzero(exact_k_path_mask, as_tuple=False)
        for u, v in pairs.tolist():
            if u < v: # Store unique, sorted pair to avoid duplicates (u,v) and (v,u)
                final_k_hop_edges_sets[k_val].add((u, v))
    
    # Convert sets to numpy arrays
    final_k_hop_edges_numpy = defaultdict(lambda: np.empty((2,0), dtype=int))
    for k_val in final_k_hop_edges_sets:
        final_k_hop_edges_numpy[k_val] = np.array(list(final_k_hop_edges_sets[k_val])).T
        
    return final_k_hop_edges_numpy


def sample_negatives_by_hop_distance(graph_nodes_in_subgraph, graph_for_path_finding, num_samples, min_distance, max_distance, specific_positive_edges_to_exclude_set, max_attempts_multiplier=200):
    """
    Generate negative examples that are within a specific hop distance range.
    Args:
        graph_nodes_in_subgraph: List of node IDs present in the current subgraph.
        graph_for_path_finding: NetworkX graph built from the edges you want to use for shortest path calculation (e.g., full subgraph 1-hop edges).
        num_samples: Number of negative samples to generate.
        min_distance: Minimum hop distance for negative samples.
        max_distance: Maximum hop distance for negative samples.
        specific_positive_edges_to_exclude_set: A set of edges (u,v) where u<v that *must not* be sampled as negatives.
        max_attempts_multiplier: Multiplier for max attempts to find negatives.
    Returns:
        Negative edges (2 x num_samples) as a numpy array.
    """
    if len(graph_nodes_in_subgraph) < 2:
        return np.empty((2, 0), dtype=int)

    negatives = []
    attempts = 0
    max_attempts = num_samples * max_attempts_multiplier
    
    nodes_for_sampling = list(graph_nodes_in_subgraph)
    
    while len(negatives) < num_samples and attempts < max_attempts:
        u, v = random.sample(nodes_for_sampling, 2)
        if u == v:
            attempts += 1
            continue
        
        u, v = (u, v) if u < v else (v, u) # Canonical representation

        # Check against specific positive edges set
        if (u, v) in specific_positive_edges_to_exclude_set:
            attempts += 1
            continue

        try:
            d = nx.shortest_path_length(graph_for_path_finding, u, v)
            # Apply the distance constraint
            if min_distance < d <= max_distance:
                negatives.append([u, v])
        except nx.NetworkXNoPath:
            # If no path exists, its distance is considered infinity.
            # Only add if max_distance is float('inf').
            if max_distance == float('inf'): 
                    negatives.append([u,v])
        
        attempts += 1
            
    return np.array(negatives).T if negatives else np.empty((2, 0), dtype=int)


def prepare_llm_training_data(pos_edges, neg_edges_pool, negative_ratio):
    """
    Create balanced dataset for LLM fine-tuning.
    Args:
        pos_edges: numpy array (2, N) of positive edges.
        neg_edges_pool: numpy array (2, M) of candidate negative edges (larger pool).
        negative_ratio: Ratio of negative to positive samples.
    Returns:
        all_edges (numpy array), all_labels (numpy array)
    """
    if pos_edges.size == 0:
        return np.empty((2,0), dtype=int), np.empty((0,), dtype=int)
        
    num_pos = pos_edges.shape[1]
    
    # Sample negative edges to match the ratio from the provided pool
    num_neg_to_sample = int(num_pos * negative_ratio)
    if neg_edges_pool.shape[1] > num_neg_to_sample:
        neg_indices = np.random.choice(neg_edges_pool.shape[1], num_neg_to_sample, replace=False)
        sampled_neg_edges = neg_edges_pool[:, neg_indices]
    else:
        sampled_neg_edges = neg_edges_pool # Use all available negatives if not enough

    pos_labels = np.ones(num_pos, dtype=int)
    neg_labels = np.zeros(sampled_neg_edges.shape[1], dtype=int)

    # Combine
    all_edges = np.concatenate([pos_edges, sampled_neg_edges], axis=1)
    all_labels = np.concatenate([pos_labels, neg_labels])

    # Shuffle
    shuffle_idx = np.random.permutation(all_edges.shape[1])
    
    return all_edges[:, shuffle_idx], all_labels[shuffle_idx]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', required=True, help="Directory to save the processed data NPZ file.")
    parser.add_argument('--dataset_name', required=True, help="Name of the dataset (e.g., 'cora', 'citeseer').")
    parser.add_argument('--text_data_file_path', required=True, help="Path to the CSV file containing node text data.")
    parser.add_argument('--graph_data_file_path', required=True, help="Path to the .pt file containing PyG graph data.")
    parser.add_argument('--node_id_field', default='node_id', help="Column name for node IDs in df_text.")
    parser.add_argument('--max_text_length', type=int, default=1000, help="Maximum length for node text (truncation).")
    parser.add_argument('--num_nodes_subgraph', type=int, default=2000, help="Target number of nodes for the subgraph.")
    parser.add_argument('--negative_ratio', type=float, default=2.0, help="Ratio of negative to positive samples for LLM training and evaluation.")
    parser.add_argument('--num_long_range_samples_total', type=int, default=300, 
                        help="Total number of long-range (e.g., 3-hop) samples to split between LLM training and k-hop testing.")
    parser.add_argument('--long_range_train_ratio', type=float, default=0.8, 
                        help="Ratio of num_long_range_samples_total for LLM training.")
    parser.add_argument('--long_range_val_ratio', type=float, default=0.1,  # New argument for validation
                        help="Ratio of num_long_range_samples_total for LLM validation (e.g., 3-hop val).")
    args = parser.parse_args()
    basics.set_seeds(42)

    # --- 1. Load and Subgraph Extraction ---
    print("Loading and extracting subgraph...")
    original_graph_data, original_df_text = load_tag_data(args.text_data_file_path, args.graph_data_file_path)
    if original_graph_data.is_directed():
        original_graph_data.edge_index = to_undirected(original_graph_data.edge_index)
    original_graph_data.edge_index, _ = coalesce(original_graph_data.edge_index, None, num_nodes=original_graph_data.num_nodes)
    original_graph_data.edge_index, _ = remove_self_loops(original_graph_data.edge_index)
    print(f"Original graph: {original_graph_data.num_nodes} nodes, {original_graph_data.edge_index.shape[1]} edges (undirected, no self-loops).")
    graph_data, df_text, node_mapping = extract_dense_subgraph(
        original_graph_data, original_df_text, args.node_id_field, args.num_nodes_subgraph
    )
    actual_num_nodes_subgraph = graph_data.num_nodes 
    print(f"Extracted subgraph has {actual_num_nodes_subgraph} nodes and {graph_data.edge_index.shape[1]} edges.")
    # Prepare node features and text
    if not hasattr(graph_data, 'x') or graph_data.x is None:
        graph_data.x = create_dummy_features(graph_data.num_nodes)
    node_texts_list = [f"Title: {row['title']} Abstract: {row['abstract']}" 
                       for _, row in df_text.iterrows()]
    node_texts = np.array(node_texts_list, dtype=object)
    # All 1-hop edges in the *full subgraph*
    all_subgraph_1_hop_edges_numpy = graph_data.edge_index.numpy()
    all_1_hop_edges_set = edges_to_set(all_subgraph_1_hop_edges_numpy)
    # Create NetworkX graph for shortest path calculations for all negative sampling 
    nx_full_subgraph = nx.Graph()
    subgraph_nodes_list = list(range(actual_num_nodes_subgraph))
    nx_full_subgraph.add_nodes_from(subgraph_nodes_list)
    nx_full_subgraph.add_edges_from(graph_data.edge_index.numpy().T.tolist())

    # --- 2. Split 1-hop edges for standard LP evaluation ---
    train_1_hop_edges, val_1_hop_edges, test_1_hop_edges = split_disjoint_edges(
        all_subgraph_1_hop_edges_numpy, train_ratio=0.8, val_ratio=0.1
    )
    print(f"1-hop splits: Train {train_1_hop_edges.shape[1]}, Val {val_1_hop_edges.shape[1]}, Test {test_1_hop_edges.shape[1]} edges.")

    # --- 3. Construct Training Graph (for long-range training) ---
    training_graph_edges_torch = torch.tensor(train_1_hop_edges, dtype=torch.long)
    training_graph_edges_torch = to_undirected(training_graph_edges_torch)
    training_graph_edges_torch, _ = coalesce(training_graph_edges_torch, None, num_nodes=actual_num_nodes_subgraph)
    training_graph_edges_torch, _ = remove_self_loops(training_graph_edges_torch)
    print(f"Training graph (1-hop train only): {training_graph_edges_torch.shape[1]} edges.")

    # --- 4. Generate 3-hop positive/negative samples for LLM training and 3-hop test ---
    print("\nGenerating long-range training and evaluation data (3-hop)...")
    all_3_hop_pos_in_train_graph = get_k_hop_edges_matrix_power(
        edge_index=training_graph_edges_torch,
        num_nodes=actual_num_nodes_subgraph,
        max_k=3,
        exclude_edges_set=None
    ).get(3, np.empty((2,0), dtype=int))
    print(f"Found {all_3_hop_pos_in_train_graph.shape[1]} 3-hop positive edges within the training graph.")
    total_3_hop_pos_to_sample = min(args.num_long_range_samples_total, all_3_hop_pos_in_train_graph.shape[1])
    hop3_train_edges = np.empty((2,0), dtype=int)
    hop3_val_edges = np.empty((2,0), dtype=int)
    hop3_test_edges = np.empty((2,0), dtype=int)
    if total_3_hop_pos_to_sample > 0:
        sampled_indices_total = np.random.choice(all_3_hop_pos_in_train_graph.shape[1], total_3_hop_pos_to_sample, replace=False)
        total_sampled_3_hop_pos = all_3_hop_pos_in_train_graph[:, sampled_indices_total]
        hop3_train_edges, hop3_val_edges, hop3_test_edges = split_disjoint_edges(
            total_sampled_3_hop_pos,
            train_ratio=args.long_range_train_ratio,
            val_ratio=args.long_range_val_ratio
        )
        print(f"  Split {total_sampled_3_hop_pos.shape[1]} 3-hop positives: Train {hop3_train_edges.shape[1]}, Val {hop3_val_edges.shape[1]}, Test {hop3_test_edges.shape[1]}.")
    else:
        print("No 3-hop positive edges found to sample for long-range tasks.")

    # --- Define GLOBAL exclusion set ---
    global_positive_exclusion_set = all_1_hop_edges_set.union(
        edges_to_set(hop3_train_edges)).union(
        edges_to_set(hop3_val_edges)).union(
        edges_to_set(hop3_test_edges)
    )
    print(f"Global positive exclusion set (1-hop + 3-hop train/val/test positives): {len(global_positive_exclusion_set)}")

    # --- Regenerate 1-hop negatives ---
    total_1_hop_neg_samples_needed = (train_1_hop_edges.shape[1] + val_1_hop_edges.shape[1] + test_1_hop_edges.shape[1]) * args.negative_ratio
    all_candidate_1_hop_neg_edges = sample_negative_edges(
        actual_num_nodes_subgraph, 
        global_positive_exclusion_set,
        num_samples = int(total_1_hop_neg_samples_needed * 1.5),
        graph_nodes_in_subgraph = subgraph_nodes_list,
        max_attempts_multiplier=500 
    )
    train_neg_1_hop_edges, val_neg_1_hop_edges, test_neg_1_hop_edges = split_disjoint_edges(
        all_candidate_1_hop_neg_edges, train_ratio=0.8, val_ratio=0.1
    )
    train_neg_1_hop_edges = train_neg_1_hop_edges[:, :int(train_1_hop_edges.shape[1] * args.negative_ratio)]
    val_neg_1_hop_edges = val_neg_1_hop_edges[:, :int(val_1_hop_edges.shape[1] * args.negative_ratio)]
    test_neg_1_hop_edges = test_neg_1_hop_edges[:, :int(test_1_hop_edges.shape[1] * args.negative_ratio)]

    # --- Generate 3-hop negative samples ---
    total_3_hop_neg_samples_needed = (hop3_train_edges.shape[1] + hop3_val_edges.shape[1] + hop3_test_edges.shape[1]) * args.negative_ratio
    all_candidate_3_hop_neg_edges = sample_negatives_by_hop_distance(
        graph_nodes_in_subgraph=subgraph_nodes_list,
        graph_for_path_finding=nx_full_subgraph,
        num_samples=int(total_3_hop_neg_samples_needed * 1.5),
        min_distance=3, max_distance=float('inf'),
        specific_positive_edges_to_exclude_set=global_positive_exclusion_set
    )
    hop3_train_neg_edges, hop3_val_neg_edges, hop3_test_neg_edges = split_disjoint_edges(
        all_candidate_3_hop_neg_edges, 
        train_ratio=args.long_range_train_ratio, 
        val_ratio=args.long_range_val_ratio
    )
    hop3_train_neg_edges = hop3_train_neg_edges[:, :int(hop3_train_edges.shape[1] * args.negative_ratio)]
    hop3_val_neg_edges = hop3_val_neg_edges[:, :int(hop3_val_edges.shape[1] * args.negative_ratio)]
    hop3_test_neg_edges = hop3_test_neg_edges[:, :int(hop3_test_edges.shape[1] * args.negative_ratio)]

    # --- Generate LLM combined training data ---
    llm_train_pos_edges = np.concatenate([train_1_hop_edges, hop3_train_edges], axis=1)
    llm_train_neg_edges_pool = np.concatenate([train_neg_1_hop_edges, hop3_train_neg_edges], axis=1)
    llm_train_edges_combined, llm_train_labels_combined = prepare_llm_training_data(
        pos_edges=llm_train_pos_edges,
        neg_edges_pool=llm_train_neg_edges_pool,
        negative_ratio=args.negative_ratio 
    )

    # --- Save final dataset ---
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    output_filename = f"{args.dataset_name}.npz"
    output_path = os.path.join(output_dir, output_filename)
    print(f"\nSaving processed data to {output_path}...")
    np.savez(
        output_path,
        # Core graph data 
        edges=graph_data.edge_index.numpy(),
        node_features=graph_data.x.numpy(),
        node_texts=node_texts,
        # Standard 1-hop splits 
        train_edges=train_1_hop_edges,
        val_edges=val_1_hop_edges,
        test_edges=test_1_hop_edges,
        train_neg_edges=train_neg_1_hop_edges,
        val_neg_edges=val_neg_1_hop_edges,
        test_neg_edges=test_neg_1_hop_edges,
        # K-hop evaluation sets (now with val)
        hop3_train_edges=hop3_train_edges, 
        hop3_val_edges=hop3_val_edges, 
        hop3_test_edges=hop3_test_edges,
        hop3_train_neg_edges=hop3_train_neg_edges, 
        hop3_val_neg_edges=hop3_val_neg_edges, 
        hop3_test_neg_edges=hop3_test_neg_edges,
        # LLM specific training data 
        llm_train_edges=llm_train_edges_combined, 
        llm_train_labels=llm_train_labels_combined, 
        # Metadata 
        node_mapping=np.array(list(node_mapping.items())) # Old_ID -> New_ID 
    )
    print("Data processing complete!")

    # --- Optional: Add sanity checks (similar to the original file) ---
    
    
if __name__ == '__main__':
    main()