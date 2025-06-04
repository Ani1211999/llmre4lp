import numpy as np
import json
import os
import argparse
import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_networkx
import networkx as nx

# Prompt template for Cora
prompt_info = {
    'Cora': {
        'BT': "Background: Academic papers with citation edges in a computer science citation network. Each paper has an abstract describing its content. Task: Predict whether a citation exists between two papers based on their abstracts.",
        'A': ". Answer template: \"Yes\" or \"No\" for training/validation/test, or a probability (0–1) for inference. Please think step by step.",
        'F': "Paper 1 abstract: ",
        'S': "Paper 2 abstract: "
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

def truncate_tokens(text, max_tokens=1000):
    tokens = text.split()
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
    return ' '.join(tokens)

def load_cora_to_npz(save_path='data/Cora/cora.npz'):
    dataset = Planetoid(root='data/Cora', name='Cora')
    data = dataset[0]
    edges = data.edge_index.cpu().numpy()
    node_labels = data.y.cpu().numpy()
    node_features = data.x.cpu().numpy()
    train_mask = data.train_mask.cpu().numpy()
    val_mask = data.val_mask.cpu().numpy()
    test_mask = data.test_mask.cpu().numpy()
    try:
        npz_data = np.load(save_path, allow_pickle=True)
        node_texts = npz_data['node_texts']
    except:
        node_texts = [f"Dummy abstract for node {i}." for i in range(data.num_nodes)]
    np.savez(
        save_path,
        edges=edges,
        node_labels=node_labels,
        node_features=node_features,
        train_masks=train_mask,
        val_masks=val_mask,
        test_masks=test_mask,
        node_texts=node_texts
    )
    return edges, node_labels, node_features, train_mask, val_mask, test_mask, node_texts

def generate_long_range_adj_matrix(adj_matrix, distances, min_distance=3):
    num_nodes = adj_matrix.shape[0]
    long_range_adj = np.zeros_like(adj_matrix)
    for u in range(num_nodes):
        for v in range(u + 1, num_nodes):
            d_uv = distances.get(u, {}).get(v, float('inf'))
            if d_uv >= min_distance:
                long_range_adj[u, v] = 1
                long_range_adj[v, u] = 1
    return long_range_adj

def generate_prompt_json(data_name, mode, save_dir):
    npz_path = f"data/{data_name}/{data_name}.npz"
    if not os.path.exists(npz_path):
        edges, node_labels, node_features, train_mask, val_mask, test_mask, text_list = load_cora_to_npz(npz_path)
    else:
        npz_data = np.load(npz_path, allow_pickle=True)
        edges = npz_data['edges']
        node_labels = npz_data['node_labels']
        node_features = npz_data['node_features']
        train_mask = npz_data['train_masks']
        val_mask = npz_data['val_masks']
        test_mask = npz_data['test_masks']
        text_list = npz_data['node_texts']

    mode_path = {
        "train_all": "train.json",
        "train_one_hop": "train.json",
        "train_two_hop": "train.json",
        "train_long_range": "train.json",
        "one_hop_infer": "one_hop_infer.json",
        "two_hop_infer": "two_hop_infer.json",
        "long_range_infer": "long_range_infer.json",
        "test_one_hop": "test.json",
        "test_two_hop": "test.json",
        "test_long_range": "test.json",
        "val_one_hop": "val.json",
        "val_two_hop": "val.json",
        "val_long_range": "val.json",
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

    if mode == "one_hop_infer":
        combined = list_result[1]
    elif mode == "two_hop_infer":
        combined = np.maximum(list_result[1], list_result[2])
    elif mode == "long_range_infer":
        combined = generate_long_range_adj_matrix(adj_matrix, distances, min_distance=3)
    elif mode == "train_all":
        combined = np.ones((num_nodes, num_nodes))
        train_nodes = np.where(train_mask)[0]
        masked_combined = np.zeros_like(combined)
        masked_combined[np.ix_(train_nodes, train_nodes)] = combined[np.ix_(train_nodes, train_nodes)]
        combined = masked_combined
    elif mode == "train_one_hop":
        combined = list_result[1]
        train_nodes = np.where(train_mask)[0]
        masked_combined = np.zeros_like(combined)
        masked_combined[np.ix_(train_nodes, train_nodes)] = combined[np.ix_(train_nodes, train_nodes)]
        combined = masked_combined
    elif mode == "train_two_hop":
        combined = np.maximum(list_result[1], list_result[2])
        train_nodes = np.where(train_mask)[0]
        masked_combined = np.zeros_like(combined)
        masked_combined[np.ix_(train_nodes, train_nodes)] = combined[np.ix_(train_nodes, train_nodes)]
        combined = masked_combined
    elif mode == "train_long_range":
        combined = generate_long_range_adj_matrix(adj_matrix, distances, min_distance=3)
        train_nodes = np.where(train_mask)[0]
        masked_combined = np.zeros_like(combined)
        masked_combined[np.ix_(train_nodes, train_nodes)] = combined[np.ix_(train_nodes, train_nodes)]
        combined = masked_combined
    elif mode == "test_one_hop":
        combined = list_result[1]
        test_nodes = np.where(test_mask)[0]
        masked_combined = np.zeros_like(combined)
        masked_combined[np.ix_(test_nodes, test_nodes)] = combined[np.ix_(test_nodes, test_nodes)]
        combined = masked_combined
    elif mode == "test_two_hop":
        combined = np.maximum(list_result[1], list_result[2])
        test_nodes = np.where(test_mask)[0]
        masked_combined = np.zeros_like(combined)
        masked_combined[np.ix_(test_nodes, test_nodes)] = combined[np.ix_(test_nodes, test_nodes)]
        combined = masked_combined
    elif mode == "test_long_range":
        combined = generate_long_range_adj_matrix(adj_matrix, distances, min_distance=3)
        test_nodes = np.where(test_mask)[0]
        masked_combined = np.zeros_like(combined)
        masked_combined[np.ix_(test_nodes, test_nodes)] = combined[np.ix_(test_nodes, test_nodes)]
        combined = masked_combined
    elif mode == "val_one_hop":
        combined = list_result[1]
        val_nodes = np.where(val_mask)[0]
        masked_combined = np.zeros_like(combined)
        masked_combined[np.ix_(val_nodes, val_nodes)] = combined[np.ix_(val_nodes, val_nodes)]
        combined = masked_combined
    elif mode == "val_two_hop":
        combined = np.maximum(list_result[1], list_result[2])
        val_nodes = np.where(val_mask)[0]
        masked_combined = np.zeros_like(combined)
        masked_combined[np.ix_(val_nodes, val_nodes)] = combined[np.ix_(val_nodes, val_nodes)]
        combined = masked_combined
    elif mode == "val_long_range":
        combined = generate_long_range_adj_matrix(adj_matrix, distances, min_distance=3)
        val_nodes = np.where(val_mask)[0]
        masked_combined = np.zeros_like(combined)
        masked_combined[np.ix_(val_nodes, val_nodes)] = combined[np.ix_(val_nodes, val_nodes)]
        combined = masked_combined
    else:
        raise ValueError(f"The mode '{mode}' is not valid.")

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
            if mode in ["train_all", "train_one_hop", "train_two_hop", "train_long_range",
                        "val_one_hop", "val_two_hop", "val_long_range",
                        "test_one_hop", "test_two_hop", "test_long_range"]:
                gpt = "Yes" if has_edge else "No"
            else:
                gpt = "Please provide a probability (0–1)."
            
            temp = {
                "id": str(id),
                "conversations": [
                    {
                        "from": "human",
                        "value": (prompt_info[data_name]['BT'] + 
                                  prompt_info[data_name]['F'] + 
                                  truncate_tokens(text_list[start_node]) + 
                                  prompt_info[data_name]['S'] + 
                                  truncate_tokens(text_list[end_node]) + 
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
    parser.add_argument("--data_name", type=str, required=True, help="Dataset name (e.g., Cora)")
    parser.add_argument("--mode", type=str, required=True, help="Mode (e.g., train_long_range, long_range_infer)")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to save JSON files")
    
    args = parser.parse_args()
    generate_prompt_json(args.data_name, args.mode, args.save_dir)