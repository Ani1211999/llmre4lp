import numpy as np
from collections import deque

def subset_amazon_npz_dense(input_path="dataset/Amazon.npz", output_path="dataset/Amazon_subset_2k_dense_filtered.npz", num_nodes=2000):
    """
    Subset the Amazon.npz file to a specified number of nodes using BFS to ensure connectivity,
    filtering node_texts to include only title and reviews.
    """
    # Load the npz file
    print(f"Loading {input_path}...")
    data = np.load(input_path, allow_pickle=True)
    
    # Build adjacency list for BFS
    edges = data['edges']
    num_total_nodes = data['node_labels'].shape[0]
    adj_list = [[] for _ in range(num_total_nodes)]
    for i in range(edges.shape[1]):
        src, dst = edges[0, i], edges[1, i]
        adj_list[src].append(dst)
        adj_list[dst].append(src)  # Undirected graph
    
    # BFS to sample a connected subset of nodes
    sampled_indices = set()
    queue = deque()
    seed = np.random.randint(0, num_total_nodes)
    queue.append(seed)
    sampled_indices.add(seed)
    
    while queue and len(sampled_indices) < num_nodes:
        node = queue.popleft()
        for neighbor in adj_list[node]:
            if neighbor not in sampled_indices:
                sampled_indices.add(neighbor)
                queue.append(neighbor)
                if len(sampled_indices) >= num_nodes:
                    break
    
    sampled_indices = list(sampled_indices)[:num_nodes]
    index_map = {old: new for new, old in enumerate(sampled_indices)}
    
    # Subset edges
    mask = np.isin(edges[0], sampled_indices) & np.isin(edges[1], sampled_indices)
    new_edges = edges[:, mask]
    new_edges[0] = np.array([index_map[i] for i in new_edges[0]])
    new_edges[1] = np.array([index_map[i] for i in new_edges[1]])
    
    # Subset other arrays
    new_node_labels = data['node_labels'][sampled_indices]
    new_node_features = data['node_features'][sampled_indices]
    new_node_texts = data['node_texts'][sampled_indices]
    
    # Filter node_texts to include only title and reviews
    filtered_node_texts = []
    for text in new_node_texts:
        # Extract title (first line)
        title_line = text.split(';')[0]  # e.g., "Name/title of the product: Candlemas: Feast of Flames"
        title = title_line.replace("Name/title of the product: ", "").strip()
        # Extract reviews (last part after "The first five reviews:")
        review_part = text.split("The first five reviews:")[-1].strip()
        filtered_text = f"{title}\nReviews: {review_part}"
        filtered_node_texts.append(filtered_text)
    
    # Create new train/val/test masks
    new_train_masks = np.zeros(num_nodes, dtype=bool)
    new_val_masks = np.zeros(num_nodes, dtype=bool)
    new_test_masks = np.zeros(num_nodes, dtype=bool)
    
    train_size = num_nodes // 2
    val_size = num_nodes // 4
    test_size = num_nodes - train_size - val_size
    shuffled_indices = np.random.permutation(num_nodes)
    train_indices = shuffled_indices[:train_size]
    val_indices = shuffled_indices[train_size:train_size + val_size]
    test_indices = shuffled_indices[train_size + val_size:]
    
    new_train_masks[train_indices] = True
    new_val_masks[val_indices] = True
    new_test_masks[test_indices] = True
    
    new_label_texts = data['label_texts']
    
    # Save the subset
    np.savez(output_path,
             edges=new_edges,
             node_labels=new_node_labels,
             node_features=new_node_features,
             train_masks=new_train_masks,
             val_masks=new_val_masks,
             test_masks=new_test_masks,
             label_texts=new_label_texts,
             node_texts=filtered_node_texts)
    
    print(f"Saved subset with {num_nodes} nodes to {output_path}")
    print(f"Number of directed edges: {new_edges.shape[1]}")
    print(f"Number of undirected edges: {new_edges.shape[1] // 2}")
    print(f"Average degree: {new_edges.shape[1] / num_nodes:.2f}")
    print(f"Number of nodes in train_masks: {np.sum(new_train_masks)}")
    print(f"Number of nodes in val_masks: {np.sum(new_val_masks)}")
    print(f"Number of nodes in test_masks: {np.sum(new_test_masks)}")

if __name__ == "__main__":
    subset_amazon_npz_dense()