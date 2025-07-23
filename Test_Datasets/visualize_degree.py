import numpy as np
import matplotlib.pyplot as plt

def plot_degree_distribution(data):
    degrees = np.zeros(data['node_features'].shape[0])
    edges = data['edges']
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
    plt.savefig('Test_Arxiv/degree_distribution_arxiv.png')
if __name__ == "__main__":
    

    data = np.load("dataset/arxiv_2023_v5.npz", allow_pickle=True)
    plot_degree_distribution(data)