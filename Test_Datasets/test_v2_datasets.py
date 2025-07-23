import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

data = np.load('dataset/Arxiv.npz')
print("Available keys:", list(data.keys()))

# Class balance
def get_balance(edges):
    return edges.shape[1]
    

def print_basic_stats(data):
    print(f"Total nodes: {data['node_features'].shape[0]}")
    print(f"Total edges: {data['edges'].shape[1]}")
    print(f"Training edges: {data['train_edges'].shape[1]}")
    print(f"Training negative edges: {data['train_neg_edges'].shape[1]}")
    print(f"Validation negative edges: {data['val_neg_edges'].shape[1]}")
    print(f"Validation edges: {data['val_edges'].shape[1]}")
    print(f"Test edges (1-hop): {data['test_edges'].shape[1]}")
    print(f"Test Negative edges (1-hop): {data['test_neg_edges'].shape[1]}")
    # print(f"2-hop test edges: {data['hop2_edges'].shape[1]}")
    # print(f"2-hop test negative edges: {data['hop2_neg_edges'].shape[1]}")
    print(f"3-hop test edges: {data['hop3_train_edges'].shape[1]}")
    print(f"3-hop negative test edges: {data['hop3_train_neg_edges'].shape[1]}")
    print(f"3-hop test edges: {data['hop3_val_edges'].shape[1]}")
    print(f"3-hop negative test edges: {data['hop3_val_neg_edges'].shape[1]}")
    print(f"3-hop test edges: {data['hop3_test_edges'].shape[1]}")
    print(f"3-hop negative test edges: {data['hop3_test_neg_edges'].shape[1]}")
    # print(f"4-hop test edges: {data['hop4_test_edges'].shape[1]}")
    # print(f"4-hop negative test edges: {data['hop4_neg_test_edges'].shape[1]}")
    
def plot_degree_distribution(edges, num_nodes):
    degrees = np.zeros(num_nodes)
    for u, v in edges.T:
        degrees[u] += 1
        degrees[v] += 1
    
    plt.figure(figsize=(10,5))
    plt.hist(degrees, bins=50, alpha=0.7)
    plt.title('Degree Distribution')
    plt.xlabel('Degree')
    plt.ylabel('Count')
    plt.yscale('log')
    plt.show()
def analyze_text_lengths(texts):
    lengths = [len(t.split()) for t in texts]
    plt.figure(figsize=(10,5))
    plt.hist(lengths, bins=50)
    plt.title('Abstract Word Count Distribution')
    plt.xlabel('Word Count')
    plt.ylabel('Frequency')
    plt.show()
    
    print(f"Max length: {max(lengths)} words")
    print(f"Min length: {min(lengths)} words")
    print(f"Average length: {np.mean(lengths):.1f} words")

def check_connectivity(edges, num_nodes):
    G = nx.Graph()
    G.add_edges_from(edges.T)
    print(f"Connected components: {nx.number_connected_components(G)}")
    print(f"Average clustering: {nx.average_clustering(G):.3f}")
    print(f"Diameter: {nx.diameter(G) if nx.is_connected(G) else 'Disconnected'}")


print_basic_stats(data)
#analyze_text_lengths(data['node_texts'])
# print("Positive train links:", data['llm_train_labels'].sum())
# print("Negative train links:", len(data['llm_train_labels']) - data['llm_train_labels'].sum())
