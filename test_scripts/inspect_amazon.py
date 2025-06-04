import numpy as np

def inspect_amazon_npz(npz_path="dataset/Amazon.npz"):
    """
    Inspect the structure of the Amazon dataset from the LLM4HeG Amazon.npz file.
    """
    # Step 1: Load the npz file
    print(f"Loading {npz_path}...")
    data = np.load(npz_path, allow_pickle=True)
    
    # Step 2: List all keys in the npz file
    print("\n=== Keys in Amazon.npz ===")
    for key in data.files:
        print(f"- {key}: Shape = {data[key].shape}, Dtype = {data[key].dtype}")
    
    # Step 3: Inspect graph structure
    # Edges
    edges = data['edges']
    num_edges_directed = edges.shape[1] if edges.ndim == 2 else len(edges)
    num_edges_undirected = num_edges_directed // 2  # Assuming directed edges are duplicated for undirected
    print(f"\n=== Graph Structure ===")
    print(f"Number of directed edges: {num_edges_directed}")
    print(f"Number of undirected edges: {num_edges_undirected}")
    
    # Nodes (infer from labels or masks)
    labels = data['node_labels'] if 'node_labels' in data.files else None
    num_nodes = len(labels) if labels is not None else data['node_features'].shape[0]
    print(f"Number of nodes: {num_nodes}")
    
    # Sample edges
    print("\nSample edges (first 5):")
    for i in range(min(5, num_edges_directed)):
        src, dst = edges[0, i], edges[1, i] if edges.ndim == 2 else (edges[i][0], edges[i][1])
        print(f"Edge {i}: ({src}, {dst})")
    
    # Step 4: Inspect labels
    if labels is not None:
        num_classes = len(np.unique(labels))
        print(f"\nNumber of classes: {num_classes}")
        print("Sample node labels (first 5):")
        for i in range(min(5, len(labels))):
            print(f"Node {i}: Label = {labels[i]}")
    
    # Step 5: Inspect label texts
    if 'label_texts' in data.files:
        label_texts = data['label_texts']
        print(f"\nLabel texts (class descriptions):")
        for i, text in enumerate(label_texts):
            print(f"Class {i}: {text}")
    
    # Step 6: Inspect node features
    if 'node_features' in data.files:
        node_features = data['node_features']
        print(f"\nNode features shape: {node_features.shape}")
        print("Sample node features (first node, first 5 dimensions):")
        print(node_features[0, :5] if node_features.ndim == 2 else node_features[0][:5])
    
    # Step 7: Inspect text attributes
    if 'node_texts' in data.files:
        node_texts = data['node_texts']
        print(f"\nNumber of node texts: {len(node_texts)}")
        print("Sample text attributes (first 5 nodes):")
        for i in range(min(5, len(node_texts))):
            print(f"Node {i}: Text = {node_texts[i]}")
    
    # Step 8: Inspect train/val/test splits
    for split in ['train_masks', 'val_masks', 'test_masks']:
        if split in data.files:
            mask = data[split]
            num_samples = np.sum(mask)
            print(f"\nNumber of nodes in {split}: {num_samples}")

if __name__ == "__main__":
    # Path to the Amazon.npz file
    npz_path = "dataset/Amazon_subset.npz"
    inspect_amazon_npz(npz_path)