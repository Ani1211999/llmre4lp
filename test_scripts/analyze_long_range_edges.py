import numpy as np
import networkx as nx
import argparse
import random
def analyze_long_range_edges(npz_path, num_long_range=150):
    data = np.load(npz_path, allow_pickle=True)
    edges = data['edges']
    
    # Build NetworkX graph
    G = nx.Graph()
    G.add_edges_from(edges.T.tolist())
    
    # Extract long-range edges (path length >= 3)
    long_range_edges = []
    nodes = set(np.unique(edges))
    for u in nodes:
        for v in nodes:
            if v <= u:
                continue
            try:
                dist = nx.shortest_path_length(G, u, v, method='dijkstra')
                if dist >= 3:
                    long_range_edges.append([u, v])
            except nx.NetworkXNoPath:
                long_range_edges.append([u, v])
    
    # Sample if needed
    if len(long_range_edges) > num_long_range:
        long_range_edges = random.sample(long_range_edges, num_long_range)
    long_range_edges = np.array(long_range_edges).T if long_range_edges else np.empty((2, 0), dtype=np.int64)
    
    # Compare with rewired edges
    rewired_edges = np.load('src/GNN/llm_pred/rewired/rewired_edges.npy')
    print(f"Generated long-range edges: {long_range_edges.shape[1]}")
    print(f"Rewired edges: {rewired_edges.shape[1]}")
    matches = set(map(tuple, long_range_edges.T)) & set(map(tuple, rewired_edges.T))
    print(f"Number of matching edges: {len(matches)}")
    print(f"Sample long-range edges: {long_range_edges[:, :5]}")
    print(f"Sample rewired edges: {rewired_edges[:, :5]}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--npz_path', required=True, help="Path to the .npz file")
    args = parser.parse_args()
    analyze_long_range_edges(args.npz_path)