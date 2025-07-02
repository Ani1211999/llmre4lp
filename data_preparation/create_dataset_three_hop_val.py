import numpy as np
from sklearn.model_selection import train_test_split
import os

def load_original_data(input_path):
    """Load the original .npz file and return as a dictionary"""
    assert os.path.exists(input_path), f"File not found: {input_path}"
    data = np.load(input_path, allow_pickle=True)
    data_dict = {key: data[key] for key in data.files}
    data.close()
    return data_dict

def split_3hop_data(test_pos, test_neg, val_ratio=0.5, random_state=42):
    """Split 3-hop test edges into validation and test sets"""
    # Convert edge arrays to correct shape if needed
    if test_pos.ndim == 2 and test_pos.shape[0] == 2:
        test_pos = test_pos.T  # Transpose to (n_edges, 2)
    if test_neg.ndim == 2 and test_neg.shape[0] == 2:
        test_neg = test_neg.T
        
    # Split positive edges
    pos_val, pos_test = train_test_split(
        test_pos,
        test_size=val_ratio,
        random_state=random_state
    )
    
    # Split negative edges
    neg_val, neg_test = train_test_split(
        test_neg,
        test_size=val_ratio,
        random_state=random_state
    )
    
    return pos_val.T, pos_test.T, neg_val.T, neg_test.T  # Return to original shape

def create_updated_dict(original_dict, pos_val, pos_test, neg_val, neg_test):
    """Create new dictionary with updated splits"""
    return {
        **original_dict,  # Start with all original data
        "hop3_val_edges": pos_val,
        "hop3_val_neg_edges": neg_val,
        "hop3_test_edges": pos_test,
        "hop3_test_neg_edges": neg_test
    }

if __name__ == "__main__":
    # Configuration
    INPUT_PATH = "dataset/cora_v2.npz"
    OUTPUT_PATH = "dataset/updated_cora_v2.npz"
    VAL_RATIO = 0.5
    RANDOM_STATE = 42
    
    # Step 1: Load original data
    original_data = load_original_data(INPUT_PATH)
    
    # Step 2: Split 3-hop test data
    pos_val, pos_test, neg_val, neg_test = split_3hop_data(
        test_pos=original_data["hop3_test_edges"],
        test_neg=original_data["hop3_test_neg_edges"],
        val_ratio=VAL_RATIO,
        random_state=RANDOM_STATE
    )
    
    # Step 3: Create updated dictionary
    updated_data = create_updated_dict(
        original_dict=original_data,
        pos_val=pos_val,
        pos_test=pos_test,
        neg_val=neg_val,
        neg_test=neg_test
    )
    
    # Step 4: Save new .npz file
    np.savez(OUTPUT_PATH, **updated_data)
    print(f"Saved updated data to {OUTPUT_PATH}")
    
    # Verification
    print("\nVerification:")
    print(f"Original 3-hop test positives: {original_data['hop3_test_edges'].shape[1]}")
    print(f"Validation positives: {pos_val.shape[1]} | Test positives: {pos_test.shape[1]}")
    print(f"Validation negatives: {neg_val.shape[1]} | Test negatives: {neg_test.shape[1]}")