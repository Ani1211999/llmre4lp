import numpy as np
import os

def edges_to_set(edges_array):
    """Convert a (2, E) numpy array of edges to a set of sorted tuples (u, v)."""
    if edges_array.size == 0:
        return set()
    return set(tuple(sorted(pair)) for pair in edges_array.T)

def set_to_edges_array(edge_set):
    """Convert a set of sorted (u, v) tuples to a (2, E) numpy array."""
    if not edge_set:
        return np.empty((2, 0), dtype=int)
    return np.array(list(edge_set)).T

def post_process_arxiv_npz(input_npz_path, output_npz_path):
    """
    Loads an NPZ file, removes specific overlapping edges from 'train_neg_edges',
    and saves the modified data to a new NPZ file.
    """
    print(f"Loading data from: {input_npz_path}")
    data = np.load(input_npz_path, allow_pickle=True)
    
    # Create a dictionary from the loaded NPZ data
    data_dict = dict(data)

    # Convert relevant arrays to sets for efficient removal
    train_neg_edges_set = edges_to_set(data_dict['train_neg_edges'])
    hop3_train_edges_set = edges_to_set(data_dict.get('hop3_train_edges', np.empty((2,0))))
    hop3_test_edges_set = edges_to_set(data_dict.get('hop3_test_edges', np.empty((2,0))))

    print(f"Original train_neg_edges count: {len(train_neg_edges_set)}")

    # Identify overlaps
    # These are the overlaps reported in your output:
    # ⚠️ Found 1 overlapping edges between 'train_neg_edges' and 'hop3_train_edges': [(31, 547)]
    # ⚠️ Found 2 overlapping edges between 'train_neg_edges' and 'hop3_test_edges': [(179, 569), (191, 254)]

    overlap_with_hop3_train = train_neg_edges_set.intersection(hop3_train_edges_set)
    overlap_with_hop3_test = train_neg_edges_set.intersection(hop3_test_edges_set)

    print(f"Detected overlaps with hop3_train_edges: {len(overlap_with_hop3_train)}")
    if overlap_with_hop3_train:
        print(f"  {list(overlap_with_hop3_train)}")
    print(f"Detected overlaps with hop3_test_edges: {len(overlap_with_hop3_test)}")
    if overlap_with_hop3_test:
        print(f"  {list(overlap_with_hop3_test)}")

    # Combine all overlaps to be removed from train_neg_edges
    edges_to_remove = overlap_with_hop3_train.union(overlap_with_hop3_test)
    
    if not edges_to_remove:
        print("No overlaps found to remove. Saving the original data to a new file.")
        np.savez(output_npz_path, **data_dict)
        print(f"Data saved to: {output_npz_path}")
        return

    print(f"Total edges to remove from train_neg_edges: {len(edges_to_remove)}")

    # Remove the overlapping edges from train_neg_edges_set
    modified_train_neg_edges_set = train_neg_edges_set - edges_to_remove
    
    # Convert the modified set back to a numpy array
    data_dict['train_neg_edges'] = set_to_edges_array(modified_train_neg_edges_set)

    print(f"New train_neg_edges count: {len(modified_train_neg_edges_set)}")
    print(f"Saving modified data to: {output_npz_path}")

    # Save the modified data to a new NPZ file
    np.savez(output_npz_path, **data_dict)
    print("Post-processing complete!")
    print(f"Original file: {input_npz_path}")
    print(f"New file with overlaps removed: {output_npz_path}")

if __name__ == "__main__":
    # Define your input and output paths
    input_file = "dataset/arxiv_2023_v8.npz" # Your current output file
    output_file = "dataset/arxiv_2023_v8_no_overlap.npz" # The new file

    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file not found at {input_file}")
    else:
        post_process_arxiv_npz(input_file, output_file)

        # Optional: Verify with your overlap checking script
        print("\nVerifying the new NPZ file for overlaps...")
        # You would typically run your `test_overlap_ds.py` script on the new file.
        # For demonstration, you might want to call its main function if it's importable.
        # Example (if test_overlap_ds.py can be imported and has a main function):
        # import test_overlap_ds
        # test_overlap_ds.main(output_file)
        print(f"Please run 'python Test_Arxiv/test_overlap_ds.py --npz_path {output_file}' to verify.")