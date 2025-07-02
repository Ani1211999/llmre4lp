import os
import sys
import numpy as np
import torch
import networkx as nx
from torch_geometric.utils import to_dense_adj, to_undirected, coalesce, remove_self_loops
from torch_geometric.data import Data
import random
import argparse
from collections import defaultdict, deque

# Assuming 'basics.py' contains set_seeds
import basics 

# Assuming data_utils.load_data is correctly set up for Cora
# You might need to adjust paths or how these are imported based on your project structure.
# For example, if data_utils is in a parent directory, you might need:
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_utils.load_data import load_graph_cora, load_text_cora


def edges_to_set(edges_array):
    """Convert a (2, E) numpy array of edges to a set of sorted tuples (u, v)."""
    if edges_array.size == 0:
        return set()
    # Ensure tuples are sorted for consistent set representation (u,v) == (v,u)
    return set(tuple(sorted(pair)) for pair in edges_array.T.tolist())

def split_disjoint_edges(all_edges_numpy, train_ratio=0.8, val_ratio=0.1):
    """
    Splits edges ensuring no overlap between train, val, test.
    Handles (u,v) and (v,u) as the same edge.
    Args:
        all_edges_numpy: numpy array of shape (2, E)
        train_ratio: Proportion of edges for the training set.
        val_ratio: Proportion of edges for the validation set.
    Returns:
        train_edges, val_edges, test_edges (numpy arrays, 2xM)
    """
    if all_edges_numpy.size == 0:
        return np.empty((2, 0), dtype=int), np.empty((2, 0), dtype=int), np.empty((2, 0), dtype=int)
    
    # Convert to set of sorted tuples to handle undirected nature
    edge_set = edges_to_set(all_edges_numpy)
    edge_list = list(edge_set)
    random.shuffle(edge_list)

    n_total = len(edge_list)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_edges = np.array(edge_list[:n_train]).T if n_train > 0 else np.empty((2,0), dtype=int)
    val_edges = np.array(edge_list[n_train:n_train + n_val]).T if n_val > 0 else np.empty((2,0), dtype=int)
    test_edges = np.array(edge_list[n_train + n_val:]).T if (n_total - n_train - n_val) > 0 else np.empty((2,0), dtype=int)
    
    return train_edges, val_edges, test_edges


def sample_negative_edges(num_nodes, positive_edges_set, num_samples, graph_nodes_list, max_attempts_multiplier=100):
    """
    Sample non-existent edges as negative examples.
    Ensures sampled negatives are not in positive_edges_set and are within graph_nodes_list.
    Args:
        num_nodes: Total number of nodes in the graph.
        positive_edges_set: A set of existing edges (u,v) where u<v.
        num_samples: How many negatives to generate.
        graph_nodes_list: List of all node IDs present in the graph.
        max_attempts_multiplier: Multiplier for max attempts to find negatives.
    Returns:
        Negative edges (2 x num_samples) as a numpy array.
    """
    if len(graph_nodes_list) < 2:
        return np.empty((2, 0), dtype=int)

    negatives = []
    attempts = 0
    max_attempts = num_samples * max_attempts_multiplier
    
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


def sample_negatives_by_hop_distance(graph_nodes_list, graph_for_path_finding, num_samples, min_distance, max_distance, specific_positive_edges_to_exclude_set, max_attempts_multiplier=200):
    """
    Generate negative examples that are within a specific hop distance range.
    Args:
        graph_nodes_list: List of all node IDs present in the graph.
        graph_for_path_finding: NetworkX graph built from the edges you want to use for shortest path calculation (e.g., full 1-hop edges).
        num_samples: Number of negative samples to generate.
        min_distance: Minimum hop distance for negative samples.
        max_distance: Maximum hop distance for negative samples.
        specific_positive_edges_to_exclude_set: A set of edges (u,v) where u<v that *must not* be sampled as negatives.
        max_attempts_multiplier: Multiplier for max attempts to find negatives.
    Returns:
        Negative edges (2 x num_samples) as a numpy array.
    """
    if len(graph_nodes_list) < 2:
        return np.empty((2, 0), dtype=int)

    negatives = []
    attempts = 0
    max_attempts = num_samples * max_attempts_multiplier
    
    while len(negatives) < num_samples and attempts < max_attempts:
        u, v = random.sample(graph_nodes_list, 2)
        if u == v:
            attempts += 1
            continue
        
        u, v = (u, v) if u < v else (v, u) # Canonical representation

        # Check against specific positive edges set
        if (u, v) in specific_positive_edges_to_exclude_set:
            attempts += 1
            continue

        try:
            dist = nx.shortest_path_length(graph_for_path_finding, source=int(u), target=int(v))
            # Apply the distance constraint
            if min_distance <= dist <= max_distance:
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
    parser = argparse.ArgumentParser(description="Preprocess Cora dataset for LLM-based link prediction.")
    parser.add_argument('--output_npz_path', type=str, default='dataset/cora_processed.npz',
                        help="Path to save the processed data NPZ file.")
    parser.add_argument('--negative_ratio', type=float, default=2.0,
                        help="Ratio of negative to positive samples for LLM training and evaluation.")
    parser.add_argument('--num_long_range_samples_total', type=int, default=300,
                        help="Total number of long-range (e.g., 3-hop) samples to split between LLM training and k-hop testing.")
    parser.add_argument('--long_range_train_ratio', type=float, default=0.4,
                        help="Ratio of total long-range samples for the training set (e.g., 3-hop train).")
    parser.add_argument('--long_range_val_ratio', type=float, default=0.3, # Added for 3-hop validation
                        help="Ratio of total long-range samples for the validation set (e.g., 3-hop val).")
    
    args = parser.parse_args()
    basics.set_seeds(42)

    print("Loading Cora dataset...")
    # Load the entire Cora graph and text data
    data, data_citeid = load_graph_cora(use_mask=False)
    text = load_text_cora(data_citeid)

    print(f"Original num of nodes: {data.num_nodes}")
    print(f"Number of texts: {len(text)}")
    print(f"Total number of edges in raw graph: {len(data.edge_index.t().tolist())}")

    # Ensure graph is undirected, coalesced, and without self-loops
    if data.is_directed():
        data.edge_index = to_undirected(data.edge_index)
    
    data.edge_index, _ = coalesce(data.edge_index, None, num_nodes=data.num_nodes)
    data.edge_index, _ = remove_self_loops(data.edge_index)

    print(f"Edges after initial cleaning: {data.edge_index.shape[1]}")
    
    # Prepare node features (Cora usually has them, but this is a safeguard)
    node_features = data.x.numpy() # Convert to numpy for saving

    # Convert node texts to numpy array
    node_texts = np.array(text, dtype=object)

    # List of all nodes in the graph (since no subgraph, it's all nodes 0 to num_nodes-1)
    graph_nodes_list = list(range(data.num_nodes))

    # All 1-hop edges in the *full graph*
    all_1_hop_edges_numpy = data.edge_index.numpy()
    all_1_hop_edges_set = edges_to_set(all_1_hop_edges_numpy)

    # Create NetworkX graph for shortest path calculations for all negative sampling
    nx_full_graph = nx.Graph()
    nx_full_graph.add_nodes_from(graph_nodes_list)
    nx_full_graph.add_edges_from(all_1_hop_edges_numpy.T.tolist()) 
    
    # --- 1. Split 1-hop edges for standard LP evaluation ---
    train_1_hop_edges, val_1_hop_edges, test_1_hop_edges = split_disjoint_edges(
        all_1_hop_edges_numpy, train_ratio=0.8, val_ratio=0.1
    )
    print(f"1-hop splits: Train {train_1_hop_edges.shape[1]}, Val {val_1_hop_edges.shape[1]}, Test {test_1_hop_edges.shape[1]} edges.")
    
    # --- 2. Construct Training Graph (for long-range training calculations) ---
    # The training graph explicitly excludes validation and test 1-hop edges 
    training_graph_edges_torch = torch.tensor(train_1_hop_edges, dtype=torch.long)
    # The following lines are commented out as the original data.edge_index was already processed for undirected, coalesce, remove_self_loops.
    # training_graph_edges_torch = to_undirected(training_graph_edges_torch)
    # training_graph_edges_torch, _ = coalesce(training_graph_edges_torch, None, num_nodes=data.num_nodes)
    # training_graph_edges_torch, _ = remove_self_loops(training_graph_edges_torch)
    print(f"Training graph (1-hop train only): {training_graph_edges_torch.shape[1]} edges.")


    # --- 3. Generate 3-hop positive/negative samples for LLM training and 3-hop test ---
    print("\nGenerating long-range training and evaluation data (3-hop)...")
    
    # Calculate ALL 3-hop positives in the training graph for initial sampling pool 
    # These are 3-hop paths that exist using only 1-hop training edges.
    all_3_hop_pos_in_train_graph = get_k_hop_edges_matrix_power(
        edge_index=training_graph_edges_torch, 
        num_nodes=data.num_nodes,
        max_k=3,
        exclude_edges_set=None 
    ).get(3, np.empty((2,0), dtype=int))
    
    print(f"Found {all_3_hop_pos_in_train_graph.shape[1]} 3-hop positive edges within the training graph.")

    # Determine how many 3-hop positives we need in total, and split 
    total_3_hop_pos_to_sample = min(args.num_long_range_samples_total, all_3_hop_pos_in_train_graph.shape[1])
    
    hop3_train_edges = np.empty((2,0), dtype=int) 
    hop3_val_edges = np.empty((2,0), dtype=int) # Added for 3-hop validation
    hop3_test_edges = np.empty((2,0), dtype=int)
    
    if total_3_hop_pos_to_sample > 0: 
        sampled_indices_total = np.random.choice(all_3_hop_pos_in_train_graph.shape[1], total_3_hop_pos_to_sample, replace=False)
        total_sampled_3_hop_pos = all_3_hop_pos_in_train_graph[:, sampled_indices_total]

        # Use split_disjoint_edges for a proper train/val/test split for 3-hop positives
        hop3_train_edges, hop3_val_edges, hop3_test_edges = split_disjoint_edges(
            total_sampled_3_hop_pos,
            train_ratio=args.long_range_train_ratio,
            val_ratio=args.long_range_val_ratio
        )
        
        print(f"  Split {total_sampled_3_hop_pos.shape[1]} 3-hop positives: Train {hop3_train_edges.shape[1]}, Val {hop3_val_edges.shape[1]}, Test {hop3_test_edges.shape[1]} edges.") 
    else:
        print("No 3-hop positive edges found to sample for long-range tasks.")

    # --- CRITICAL FIX: Define the GLOBAL exclusion set *after* all positive sets are determined ---
    # This set will contain ALL 1-hop positive edges (train, val, test from original graph split)
    # AND ALL 3-hop positive edges (train, val, and test for the 3-hop task, sampled from training graph paths).
    global_positive_exclusion_set = all_1_hop_edges_set.union(
        edges_to_set(hop3_train_edges)).union(
        edges_to_set(hop3_val_edges)).union( # Added 3-hop validation edges
        edges_to_set(hop3_test_edges)
    )
    print(f"Global positive exclusion set (1-hop + 3-hop train/val/test positives): {len(global_positive_exclusion_set)}")


    # --- Regenerate 1-hop negatives using the GLOBAL exclusion set ---
    total_1_hop_neg_samples_needed = (train_1_hop_edges.shape[1] + val_1_hop_edges.shape[1] + test_1_hop_edges.shape[1]) * args.negative_ratio
    
    all_candidate_1_hop_neg_edges = sample_negative_edges(
        data.num_nodes, 
        global_positive_exclusion_set, # Use the comprehensive GLOBAL exclusion set
        num_samples = int(total_1_hop_neg_samples_needed * 1.5), # Generate more than needed to ensure enough are found
        graph_nodes_list = graph_nodes_list,
        max_attempts_multiplier=500 # Increased attempts for robustness
    )
    print(f"Re-generated {all_candidate_1_hop_neg_edges.shape[1]} total candidate 1-hop negative edges using global exclusion.")

    # Split the candidate 1-hop negatives disjoinly
    train_neg_1_hop_edges, val_neg_1_hop_edges, test_neg_1_hop_edges = split_disjoint_edges(
        all_candidate_1_hop_neg_edges, train_ratio=0.8, val_ratio=0.1
    )
    # Adjust number of negatives to match ratio, by truncating if too many were sampled
    train_neg_1_hop_edges = train_neg_1_hop_edges[:, :int(train_1_hop_edges.shape[1] * args.negative_ratio)]
    val_neg_1_hop_edges = val_neg_1_hop_edges[:, :int(val_1_hop_edges.shape[1] * args.negative_ratio)]
    test_neg_1_hop_edges = test_neg_1_hop_edges[:, :int(test_1_hop_edges.shape[1] * args.negative_ratio)]
    print(f"1-hop negative splits (re-adjusted): Train {train_neg_1_hop_edges.shape[1]}, Val {val_neg_1_hop_edges.shape[1]}, Test {test_neg_1_hop_edges.shape[1]} edges.")


    # --- Generate 3-hop negative samples using the GLOBAL exclusion set ---
    total_3_hop_neg_samples_needed = (hop3_train_edges.shape[1] + hop3_val_edges.shape[1] + hop3_test_edges.shape[1]) * args.negative_ratio

    all_candidate_3_hop_neg_edges = sample_negatives_by_hop_distance(
        graph_nodes_list = graph_nodes_list,
        graph_for_path_finding=nx_full_graph, # Use full graph for distances
        num_samples=int(total_3_hop_neg_samples_needed * 1.5), # Generate more than needed
        min_distance=3, max_distance=float('inf'), # Negatives must have distance >= 3 (including infinite)
        specific_positive_edges_to_exclude_set=global_positive_exclusion_set, # Use the comprehensive GLOBAL exclusion set
        max_attempts_multiplier=500 # Increased attempts for robustness
    )
    print(f"Generated {all_candidate_3_hop_neg_edges.shape[1]} total candidate 3-hop negative edges using global exclusion.")

    # Split the candidate 3-hop negatives disjointly
    hop3_train_neg_edges, hop3_val_neg_edges, hop3_test_neg_edges = split_disjoint_edges( 
        all_candidate_3_hop_neg_edges, 
        train_ratio=args.long_range_train_ratio, 
        val_ratio=args.long_range_val_ratio
    )
    # Adjust number of negatives to match ratio, by truncating if too many were sampled
    hop3_train_neg_edges = hop3_train_neg_edges[:, :int(hop3_train_edges.shape[1] * args.negative_ratio)]
    hop3_val_neg_edges = hop3_val_neg_edges[:, :int(hop3_val_edges.shape[1] * args.negative_ratio)] # Adjusted for validation
    hop3_test_neg_edges = hop3_test_neg_edges[:, :int(hop3_test_edges.shape[1] * args.negative_ratio)]
    print(f"3-hop negative splits (re-adjusted): Train {hop3_train_neg_edges.shape[1]}, Val {hop3_val_neg_edges.shape[1]}, Test {hop3_test_neg_edges.shape[1]}.")


    # --- Generate LLM combined training data (1-hop train + 3-hop train) ---
    print("\nPreparing LLM training data (1-hop and 3-hop combined)...")
    # Combine all positive and negative edges designated for LLM training
    llm_train_pos_edges = np.concatenate([train_1_hop_edges, hop3_train_edges], axis=1)
    llm_train_neg_edges_pool = np.concatenate([train_neg_1_hop_edges, hop3_train_neg_edges], axis=1)

    llm_train_edges_combined, llm_train_labels_combined = prepare_llm_training_data(
        pos_edges=llm_train_pos_edges,
        neg_edges_pool=llm_train_neg_edges_pool,
        negative_ratio=args.negative_ratio 
    )
    print(f"LLM combined training data: {llm_train_edges_combined.shape[1]} samples ({np.sum(llm_train_labels_combined)} pos, {np.sum(llm_train_labels_combined==0)} neg).")


    # --- Save final dataset ---
    print(f"\nSaving processed data to {args.output_npz_path}...")
    np.savez(
        args.output_npz_path,
        # Core graph data 
        edges=all_1_hop_edges_numpy, # All 1-hop edges of the full graph after cleaning
        node_features=node_features,
        node_texts=node_texts,
        
        # Standard 1-hop splits 
        train_edges=train_1_hop_edges,
        val_edges=val_1_hop_edges,
        test_edges=test_1_hop_edges,
        train_neg_edges=train_neg_1_hop_edges,
        val_neg_edges=val_neg_1_hop_edges,
        test_neg_edges=test_neg_1_hop_edges,
        
        # K-hop evaluation/training sets (only 3-hop now) 
        hop3_train_edges=hop3_train_edges, 
        hop3_val_edges=hop3_val_edges, # Added 3-hop validation positives
        hop3_test_edges=hop3_test_edges,
        hop3_train_neg_edges=hop3_train_neg_edges, 
        hop3_val_neg_edges=hop3_val_neg_edges, # Added 3-hop validation negatives
        hop3_test_neg_edges=hop3_test_neg_edges,
        
        # LLM specific training data (1-hop train + 3-hop train combined) 
        llm_train_edges=llm_train_edges_combined, 
        llm_train_labels=llm_train_labels_combined
    )
    print("Data processing complete!")
    loaded_data = np.load(args.output_npz_path, allow_pickle=True)
    
    train_edges_set = edges_to_set(loaded_data['train_edges'])
    val_edges_set = edges_to_set(loaded_data['val_edges'])
    test_edges_set = edges_to_set(loaded_data['test_edges'])
    train_neg_set = edges_to_set(loaded_data['train_neg_edges'])
    val_neg_set = edges_to_set(loaded_data['val_neg_edges'])
    test_neg_set = edges_to_set(loaded_data['test_neg_edges'])
    hop3_train_set = edges_to_set(loaded_data['hop3_train_edges'])
    hop3_train_neg_set = edges_to_set(loaded_data['hop3_train_neg_edges'])
    hop3_val_set = edges_to_set(loaded_data['hop3_val_edges'])
    hop3_val_neg_set = edges_to_set(loaded_data['hop3_val_neg_edges'])

    hop3_test_set = edges_to_set(loaded_data['hop3_test_edges'])
    hop3_test_neg_set = edges_to_set(loaded_data['hop3_test_neg_edges'])
    
    # --- Core disjointness checks (Positives vs Negatives, within splits) ---
    print("\n--- 1-Hop Split Disjointness ---")
    print(f"Overlap train_edges vs val_edges: {len(train_edges_set.intersection(val_edges_set))}")
    print(f"Overlap train_edges vs test_edges: {len(train_edges_set.intersection(test_edges_set))}")
    print(f"Overlap val_edges vs test_edges: {len(val_edges_set.intersection(test_edges_set))}")

    print("\n--- 3-Hop Split Disjointness ---")
    print(f"Overlap hop3_train_edges vs hop3_test_edges: {len(hop3_train_set.intersection(hop3_test_set))}")
    print(f"Overlap hop3_train_edges vs hop3_val_edges: {len(hop3_train_set.intersection(hop3_val_set))}")
    print(f"Overlap hop3_val_edges vs hop3_test_edges: {len(hop3_val_set.intersection(hop3_test_set))}")
    print(f"Overlap hop3_train_neg_edges vs hop3_test_neg_edges: {len(hop3_train_neg_set.intersection(hop3_test_neg_set))}")
    print(f"Overlap hop3_train_neg_edges vs hop3_val_neg_edges: {len(hop3_train_neg_set.intersection(hop3_val_neg_set))}")
    print(f"Overlap hop3_val_neg_edges vs hop3_test_neg_edges: {len(hop3_val_neg_set.intersection(hop3_test_neg_set))}")

    print("\n--- Negative vs Positive Overlaps (CRITICAL) ---")
    # Overlap between 1-hop negatives and ANY positive (1-hop or 3-hop)
    all_positive_edges_final_set = train_edges_set.union(val_edges_set).union(test_edges_set).union(hop3_train_set).union(hop3_val_set).union(hop3_test_set)
    
    overlap_train_neg_pos = train_neg_set.intersection(all_positive_edges_final_set)
    if overlap_train_neg_pos:
        print(f"⚠️ Overlap between 'train_neg_edges' and ANY positive edge: {len(overlap_train_neg_pos)}")
        print(list(overlap_train_neg_pos)[:5])
    else:
        print("✅ No overlap between 'train_neg_edges' and ANY positive edge.")

    overlap_val_neg_pos = val_neg_set.intersection(all_positive_edges_final_set)
    if overlap_val_neg_pos:
        print(f"⚠️ Overlap between 'val_neg_edges' and ANY positive edge: {len(overlap_val_neg_pos)}")
        print(list(overlap_val_neg_pos)[:5])
    else:
        print("✅ No overlap between 'val_neg_edges' and ANY positive edge.")

    overlap_test_neg_pos = test_neg_set.intersection(all_positive_edges_final_set)
    if overlap_test_neg_pos:
        print(f"⚠️ Overlap between 'test_neg_edges' and ANY positive edge: {len(overlap_test_neg_pos)}")
        print(list(overlap_test_neg_pos)[:5])
    else:
        print("✅ No overlap between 'test_neg_edges' and ANY positive edge.")

    overlap_hop3_train_neg_pos = hop3_train_neg_set.intersection(all_positive_edges_final_set)
    if overlap_hop3_train_neg_pos:
        print(f"⚠️ Overlap between 'hop3_train_neg_edges' and ANY positive edge: {len(overlap_hop3_train_neg_pos)}")
        print(list(overlap_hop3_train_neg_pos)[:5])
    else:
        print("✅ No overlap between 'hop3_train_neg_edges' and ANY positive edge.")

    overlap_hop3_val_neg_pos = hop3_val_neg_set.intersection(all_positive_edges_final_set)
    if overlap_hop3_train_neg_pos:
        print(f"⚠️ Overlap between 'hop3_val_neg_edges' and ANY positive edge: {len(overlap_hop3_val_neg_pos)}")
        print(list(overlap_hop3_val_neg_pos)[:5])
    else:
        print("✅ No overlap between 'hop3_train_neg_edges' and ANY positive edge.")
        
    overlap_hop3_test_neg_pos = hop3_test_neg_set.intersection(all_positive_edges_final_set)
    if overlap_hop3_test_neg_pos:
        print(f"⚠️ Overlap between 'hop3_test_neg_edges' and ANY positive edge: {len(overlap_hop3_test_neg_pos)}")
        print(list(overlap_hop3_test_neg_pos)[:5])
    else:
        print("✅ No overlap between 'hop3_test_neg_edges' and ANY positive edge.")

    print("\n--- Finished all sanity checks ---")

if __name__ == "__main__":
    main()
