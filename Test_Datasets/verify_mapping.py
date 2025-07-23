import numpy as np

def verify_node_mapping(data):
    node_mapping = dict(data['node_mapping'])
    original_ids = set(node_mapping.keys())
    new_ids = set(node_mapping.values())
    
    print("Unique original node IDs mapped:", len(original_ids))
    print("Unique new node IDs assigned:", len(new_ids))
    assert len(original_ids) == len(new_ids), "Mapping not injective!"
    assert max(new_ids) == len(new_ids) - 1, "New node IDs are not contiguous!"

def verify_edges_in_range(data, num_nodes):
    for key in ['edges', 'train_edges', 'val_edges', 'test_edges',
                'hop2_edges', 'hop3_test_edges', 'hop4_test_edges']:
        if key in data:
            edges = data[key]
            invalid = [(u, v) for u, v in edges.T if u >= num_nodes or v >= num_nodes]
            if invalid:
                print(f"Found invalid edges in '{key}':", invalid[:5], "...")
            else:
                print(f"All edges in '{key}' are valid.")

if __name__ == "__main__":
    data = np.load("dataset/arxiv_2023_v5.npz", allow_pickle=True)
    num_nodes = data['node_features'].shape[0]

    verify_node_mapping(data)
    verify_edges_in_range(data, num_nodes)