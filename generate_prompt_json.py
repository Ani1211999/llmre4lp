import numpy as np
import json
import os
import argparse
import networkx as nx

# Prompt template for Arxiv
prompt_info = {
    'Arxiv': {
        'BT': "Background: Papers in the arXiv 2023 network. Each paper is described by its title and abstract. Task: Predict whether a citation relationship exists between two papers based on their descriptions.",
        'A': ". Answer template: \"Yes\" or \"No\" for training/validation/test, or a probability (0–1) for inference. Please think step by step.",
        'F': "Paper 1 description: ",
        'S': "Paper 2 description: "
    },
}

def load_arxiv_npz(npz_path):
    """
    Load the preprocessed Arxiv 2023 .npz file with node_texts and split edges.
    """
    data = np.load(npz_path, allow_pickle=True)
    edges = data['edges']
    train_edges = data['train_edges']
    val_edges = data['val_edges']
    test_edges = data['test_edges']
    train_neg_edges = data['train_neg_edges']
    val_neg_edges = data['val_neg_edges']
    test_neg_edges = data['test_neg_edges']
    long_range_edges = data['long_range_edges']
    long_range_neg_edges = data['long_range_neg_edges']
    node_labels = data['node_labels']
    node_features = data['node_features']
    node_texts = data['node_texts']
    label_texts = data['label_texts']
    return (edges, train_edges, val_edges, test_edges, train_neg_edges, val_neg_edges, test_neg_edges,
            long_range_edges, long_range_neg_edges, node_labels, node_features, node_texts, label_texts)

def generate_prompt_json(data_name, mode, save_dir, npz_path):
    # Load the Arxiv dataset
    (edges, train_edges, val_edges, test_edges, train_neg_edges, val_neg_edges, test_neg_edges,
     long_range_edges, long_range_neg_edges, node_labels, node_features, node_texts, label_texts) = load_arxiv_npz(npz_path)

    mode_path = {
        "train_all": "train_all.json",
        "val_all": "val_all.json",
        "test_all": "test_all.json",
        "long_range_infer": "long_range_infer.json",
    }

    # Select edge pairs based on mode
    if mode == "train_all":
        pos_edges = train_edges
        neg_edges = train_neg_edges
    elif mode == "val_all":
        pos_edges = val_edges
        neg_edges = val_neg_edges
    elif mode == "test_all":
        pos_edges = test_edges
        neg_edges = test_neg_edges
    elif mode == "long_range_infer":
        pos_edges = long_range_edges  # Use long-range edges (distance >= 3)
        neg_edges = long_range_neg_edges
    else:
        raise ValueError(f"The mode '{mode}' is not valid. Use 'train_all', 'val_all', 'test_all', or 'long_range_infer'.")

    # Combine positive and negative edges
    prompt_list = []
    id = 0

    # Add positive edges
    for i in range(pos_edges.shape[1]):
        start_node, end_node = pos_edges[0, i], pos_edges[1, i]
        gpt = "Yes" if mode in ["train_all", "val_all", "test_all"] else "Please provide a probability (0–1)."
        ground_truth = "Yes"  # Ground truth for positive edges
        temp = {
            "id": f"{start_node}_{end_node}",
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
            ],
            "ground_truth": ground_truth  # Added for validation
        }
        prompt_list.append(temp)
        id += 1

    # Add negative edges
    for i in range(neg_edges.shape[1]):
        start_node, end_node = neg_edges[0, i], neg_edges[1, i]
        gpt = "No" if mode in ["train_all", "val_all", "test_all"] else "Please provide a probability (0–1)."
        ground_truth = "No"  # Ground truth for negative edges
        temp = {
            "id": f"{start_node}_{end_node}",
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
            ],
            "ground_truth": ground_truth  # Added for validation
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
    parser.add_argument("--data_name", type=str, default="Arxiv", help="Dataset name (e.g., Arxiv)")
    parser.add_argument("--mode", type=str, required=True, help="Mode (train_all, val_all, test_all, or long_range_infer)")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to save JSON files")
    parser.add_argument("--npz_path", type=str, default="dataset/arxiv_2023.npz", help="Path to the .npz file")
    
    args = parser.parse_args()
    generate_prompt_json(args.data_name, args.mode, args.save_dir, args.npz_path)