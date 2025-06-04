import numpy as np
import json
import os
import argparse
import networkx as nx

# Prompt template for Amazon
prompt_info = {
    'Amazon': {
        'BT': "Background: Products in an Amazon co-purchasing network. Each product is described by its title and review summaries. Task: Predict whether two products are frequently bought together based on their descriptions.",
        'A': ". Answer template: \"Yes\" or \"No\" for training/validation/test, or a probability (0–1) for inference. Please think step by step.",
        'F': "Product 1 description: ",
        'S': "Product 2 description: "
    },
}

def nhoodSplit(adj: np.ndarray, nhood):
    assert adj.ndim == 2 and adj.shape[0] == adj.shape[1]
    if np.isnan(nhood):
        return np.ones(adj.shape)
    mt = np.eye(adj.shape[1])
    mtList = [mt]
    i = 0
    edge_sum = 0
    while i < nhood:
        prev_mt = mt
        mt = mt @ (adj + np.eye(adj.shape[0]))
        mt = (mt > 0).astype(mt.dtype)
        new_edge_sum = np.sum(mt)
        if edge_sum == new_edge_sum:
            break
        else:
            edge_sum = new_edge_sum
        i += 1
        mtList.append(mt - prev_mt)
    return mtList

def load_amazon_npz(npz_path):
    """
    Load the preprocessed Amazon subset .npz file with filtered node_texts.
    """
    data = np.load(npz_path, allow_pickle=True)
    edges = data['edges']
    node_labels = data['node_labels']
    node_features = data['node_features']
    train_masks = data['train_masks']
    val_masks = data['val_masks']
    test_masks = data['test_masks']
    node_texts = data['node_texts']
    label_texts = data['label_texts']
    return edges, node_labels, node_features, train_masks, val_masks, test_masks, node_texts, label_texts

def generate_long_range_adj_matrix(adj_matrix, distances, min_distance=3, sample_ratio=0.1):
    num_nodes = adj_matrix.shape[0]
    long_range_adj = np.zeros_like(adj_matrix)
    long_range_indices = []
    for u in range(num_nodes):
        for v in range(u + 1, num_nodes):
            d_uv = distances.get(u, {}).get(v, float('inf'))
            if d_uv >= min_distance:
                long_range_indices.append((u, v))
    sample_size = int(len(long_range_indices) * sample_ratio)
    if sample_size > 0:
        sampled_indices = np.random.choice(len(long_range_indices), sample_size, replace=False)
        for idx in sampled_indices:
            u, v = long_range_indices[idx]
            long_range_adj[u, v] = 1
            long_range_adj[v, u] = 1
    return long_range_adj

def generate_prompt_json(data_name, mode, save_dir, npz_path):
    # Load the Amazon dataset
    edges, node_labels, node_features, train_masks, val_masks, test_masks, node_texts, label_texts = load_amazon_npz(npz_path)

    mode_path = {
        "train_all": "train_all.json",
        "val_all": "val_all.json",
        "test_all": "test_all.json",
        "long_range_infer": "long_range_infer.json",
    }

    num_nodes = len(node_labels)
    adj_matrix = np.zeros((num_nodes, num_nodes))
    for i in range(edges.shape[1]):
        start_node = edges[0, i]
        end_node = edges[1, i]
        adj_matrix[start_node, end_node] = 1
        adj_matrix[end_node, start_node] = 1

    G_nx = nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph)
    distances = dict(nx.all_pairs_shortest_path_length(G_nx))

    list_result = nhoodSplit(adj_matrix, nhood=2)
    nonzero_counts = [np.count_nonzero(matrix) for matrix in list_result]
    print("Number of non-zero elements in the one-hop and two-hop adjacency matrix:", nonzero_counts)

    # Adjust sample ratio for train_all to reduce size
    train_all_sample_ratio = 0.01 if mode == "train_all" else 0.1

    if mode == "long_range_infer":
        combined = generate_long_range_adj_matrix(adj_matrix, distances, sample_ratio=train_all_sample_ratio)
    elif mode in ["train_all", "val_all", "test_all"]:
        combined = np.maximum(list_result[1], list_result[2])  # Two-hop neighborhood
        # Use pre-existing masks to filter edges
        edge_mask = None
        if mode == "train_all":
            edge_mask = train_masks
        elif mode == "val_all":
            edge_mask = val_masks
        elif mode == "test_all":
            edge_mask = test_masks
        
        if edge_mask is not None:
            combined = np.zeros_like(combined)
            for u in range(num_nodes):
                for v in range(u + 1, num_nodes):
                    if edge_mask[u] or edge_mask[v]:  # Include edges involving masked nodes
                        combined[u, v] = 1
                        combined[v, u] = 1
        combined += generate_long_range_adj_matrix(adj_matrix, distances, sample_ratio=train_all_sample_ratio)
    else:
        raise ValueError(f"The mode '{mode}' is not valid for quick output generation. Use 'train_all', 'val_all', 'test_all', or 'long_range_infer'.")

    # Ensure symmetry
    combined = combined + combined.T

    nonzero_count = np.count_nonzero(combined)
    print("Number of non-zero elements in the target template matrix:", nonzero_count)
    is_symmetric = np.array_equal(combined, combined.T)
    print("Is the target template matrix symmetric:", is_symmetric)

    prompt_list = []
    id = 0
    for start_node in range(num_nodes):
        for end_node in range(start_node + 1, num_nodes):
            if combined[start_node, end_node] <= 0:
                continue
            has_edge = G_nx.has_edge(start_node, end_node) or G_nx.has_edge(end_node, start_node)
            if mode in ["train_all", "val_all", "test_all"]:
                gpt = "Yes" if has_edge else "No"
            else:
                gpt = "Please provide a probability (0–1)."
            
            temp = {
                "id": f"{start_node}_{end_node}",  # Changed to u_v format
                "conversations": [
                    {
                        "from": "human",
                        "value": (prompt_info[data_name]['BT'] + 
                                  prompt_info[data_name]['F'] + 
                                  node_texts[start_node] + 
                                  prompt_info[data_name]['S'] + 
                                  node_texts[end_node] + 
                                  prompt_info[data_name]['A'])
                    },
                    {"from": "gpt", "value": gpt}
                ]
            }
            prompt_list.append(temp)
            id += 1
    
    print("Template length:", len(prompt_list))
    print("First template:", prompt_list[0] if prompt_list else "Empty")
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, mode_path[mode])
    print("Saving to:", save_path)
    with open(save_path, "w") as fout:
        json.dump(prompt_list, fout, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", type=str, default="Amazon", help="Dataset name (e.g., Amazon)")
    parser.add_argument("--mode", type=str, required=True, help="Mode (train_all, val_all, test_all, or long_range_infer)")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to save JSON files")
    parser.add_argument("--npz_path", type=str, default="dataset/Amazon_subset_2k_dense_filtered.npz", help="Path to the .npz file")
    
    args = parser.parse_args()
    generate_prompt_json(args.data_name, args.mode, args.save_dir, args.npz_path)