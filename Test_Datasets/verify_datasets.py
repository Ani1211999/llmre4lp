import numpy as np

def verify_npz_structure(filepath):
    """Verify all keys and split sizes in the updated .npz file"""
    data = np.load(filepath, allow_pickle=True)
    
    print("="*50)
    print(f"Verifying: {filepath}")
    print("="*50)
    
    # 1. Check all expected keys exist
    expected_keys = {
        # Core
        'edges', 'node_features', 'node_texts', 'node_mapping',
        # 1-hop splits
        'train_edges', 'val_edges', 'test_edges',
        'train_neg_edges', 'val_neg_edges', 'test_neg_edges',
        # 3-hop splits
        'hop3_train_edges', 'hop3_train_neg_edges',
        'hop3_val_edges', 'hop3_val_neg_edges',
        'hop3_test_edges', 'hop3_test_neg_edges',
        # LLM
        'llm_train_edges', 'llm_train_labels'
    }
    
    missing_keys = expected_keys - set(data.files)
    extra_keys = set(data.files) - expected_keys
    
    print("\n[1/3] Key Verification:")
    print(f"Missing keys: {missing_keys or 'None'}")
    print(f"Extra keys: {extra_keys or 'None'}")
    
    # 2. Verify split sizes
    print("\n[2/3] Split Size Verification:")
    orig_pos = data['hop3_test_edges'].shape[1] + data['hop3_val_edges'].shape[1]
    orig_neg = data['hop3_test_neg_edges'].shape[1] + data['hop3_val_neg_edges'].shape[1]
    
    print(f"3-hop positives | Original: {orig_pos} = Val: {data['hop3_val_edges'].shape[1]} + Test: {data['hop3_test_edges'].shape[1]}")
    print(f"3-hop negatives | Original: {orig_neg} = Val: {data['hop3_val_neg_edges'].shape[1]} + Test: {data['hop3_test_neg_edges'].shape[1]}")
    
    # Verify no overlap between validation and test edges
    val_set = set(tuple(e) for e in data['hop3_val_edges'].T)
    test_set = set(tuple(e) for e in data['hop3_test_edges'].T)
    assert len(val_set & test_set) == 0, "Overlap between val/test sets!"
    # 3. Sample edge checks
    print("\n[3/3] Sample Edge Verification:")
    print("First validation positive edge:", data['hop3_val_edges'][:,0])
    print("First test negative edge:", data['hop3_test_neg_edges'][:,0])
    print("First Training positive edge:", data['train_edges'][:,0])
    data.close()

if __name__ == "__main__":
    verify_npz_structure("dataset/updated_cora_v2.npz")