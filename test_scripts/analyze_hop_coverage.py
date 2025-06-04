import numpy as np
import networkx as nx
import argparse
import os

def load_npz_data(npz_path):
    """Load data from the .npz file."""
    data = np.load(npz_path, allow_pickle=True)
    edges = data['edges']
    train_edges = data['train_edges']
    test_edges = data['test_edges']
    return edges, train_edges, test_edges

def analyze_hop_coverage(npz_path):
    # Load data
    edges, train_edges, test_edges = load_npz_data(npz_path)
    
    # Build NetworkX graph from original edges
    G = nx.Graph()
    G.add_edges_from(edges.T.tolist())
    num_nodes = G.number_of_nodes()
    
    # Function to compute hop distance for an edge
    def get_hop_distance(u, v):
        try:
            return nx.shortest_path_length(G, u, v)
        except nx.NetworkXNoPath:
            return float('inf')  # Disconnected nodes

    # Analyze train edges
    train_hops = [get_hop_distance(u, v) for u, v in train_edges.T]
    train_1hop = sum(1 for h in train_hops if h == 1)
    train_2hop = sum(1 for h in train_hops if h == 2)
    train_3hop_plus = sum(1 for h in train_hops if h >= 3)
    train_disconnected = sum(1 for h in train_hops if h == float('inf'))
    print(f"\nTrain Edges (total: {len(train_hops)}):")
    print(f"1-hop: {train_1hop} ({train_1hop/len(train_hops)*100:.1f}%)")
    print(f"2-hop: {train_2hop} ({train_2hop/len(train_hops)*100:.1f}%)")
    print(f"3+ hop: {train_3hop_plus} ({train_3hop_plus/len(train_hops)*100:.1f}%)")
    print(f"Disconnected: {train_disconnected} ({train_disconnected/len(train_hops)*100:.1f}%)")

    # Analyze test edges
    test_hops = [get_hop_distance(u, v) for u, v in test_edges.T]
    test_1hop = sum(1 for h in test_hops if h == 1)
    test_2hop = sum(1 for h in test_hops if h == 2)
    test_3hop_plus = sum(1 for h in test_hops if h >= 3)
    test_disconnected = sum(1 for h in test_hops if h == float('inf'))
    print(f"\nTest Edges (total: {len(test_hops)}):")
    print(f"1-hop: {test_1hop} ({test_1hop/len(test_hops)*100:.1f}%)")
    print(f"2-hop: {test_2hop} ({test_2hop/len(test_hops)*100:.1f}%)")
    print(f"3+ hop: {test_3hop_plus} ({test_3hop_plus/len(test_hops)*100:.1f}%)")
    print(f"Disconnected: {test_disconnected} ({test_disconnected/len(test_hops)*100:.1f}%)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz_path", type=str, default="dataset/arxiv_2023.npz", help="Path to the .npz file")
    args = parser.parse_args()
    analyze_hop_coverage(args.npz_path)