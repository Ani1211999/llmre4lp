import numpy as np
import torch
import dgl
import os

def get_dataset_stats(npz_path, rewired_edges_path):
    """
    Loads the dataset and prints basic statistics.

    Args:
        npz_path (str): Path to the .npz file containing node features and edge splits.
        rewired_edges_path (str): Path to the .npy file containing rewired edges.
    """
    print(f"Attempting to load data from: {npz_path}")
    print(f"Attempting to load rewired edges from: {rewired_edges_path}")

    # Check if files exist
    if not os.path.exists(npz_path):
        print(f"Error: .npz file not found at {npz_path}")
        return
    if not os.path.exists(rewired_edges_path):
        print(f"Error: Rewired edges file not found at {rewired_edges_path}")
        return

    try:
        data = np.load(npz_path, allow_pickle=True)
        x = torch.tensor(data['node_features'], dtype=torch.float)
        train_edges = torch.tensor(data['train_edges'], dtype=torch.long)
        train_neg_edges = torch.tensor(data['train_neg_edges'], dtype=torch.long)
        test_3hop_edges = torch.tensor(data['hop3_test_edges'], dtype=torch.long)
        test_3hop_neg_edges = torch.tensor(data['hop3_test_neg_edges'], dtype=torch.long)

        rewired_edges = torch.tensor(np.load(rewired_edges_path), dtype=torch.long)

        # Combine training and rewired edges for the graph structure
        # Also add reverse edges to make the graph undirected for GNN processing
        edge_index = torch.cat([train_edges, rewired_edges], dim=1)
        edge_index = torch.cat([edge_index, edge_index[[1, 0]]], dim=1)

        num_nodes = x.size(0)
        num_features = x.size(1)
        num_train_edges = train_edges.size(1)
        num_train_neg_edges = train_neg_edges.size(1)
        num_test_3hop_edges = test_3hop_edges.size(1)
        num_test_3hop_neg_edges = test_3hop_neg_edges.size(1)
        num_rewired_edges = rewired_edges.size(1)
        num_total_graph_edges = edge_index.size(1)

        print("\n--- Dataset Statistics ---")
        print(f"Number of Nodes: {num_nodes}")
        print(f"Number of Node Features: {num_features}")
        print(f"Number of Training Edges (Positive): {num_train_edges}")
        print(f"Number of Training Negative Edges: {num_train_neg_edges}")
        print(f"Number of Rewired Edges: {num_rewired_edges}")
        print(f"Number of 3-hop Test Edges (Positive): {num_test_3hop_edges}")
        print(f"Number of 3-hop Test Negative Edges: {num_test_3hop_neg_edges}")
        print(f"Total Edges in Graph (Train + Rewired + Reverse): {num_total_graph_edges}")
        print(f"Graph Density (approx): {num_total_graph_edges / (num_nodes * (num_nodes - 1)):.6f}") # For undirected, no self-loops

    except Exception as e:
        print(f"An error occurred while loading data: {e}")

if __name__ == "__main__":
    # Define the paths to your dataset files
    # Adjust these paths if your script is run from a different directory
    npz_file_path = "dataset/cora_final_dataset.npz"
    rewired_edges_file_path = "src/GNN/cora_rewired_edges/rewired_edges.npy"

    get_dataset_stats(npz_file_path, rewired_edges_file_path)
