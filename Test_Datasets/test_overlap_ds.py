import numpy as np

def edges_to_set(edges_array):
    """Convert a (2, E) numpy array of edges to a set of sorted tuples (u, v)."""
    if edges_array.size == 0:
        return set()
    # Ensure tuples are sorted for consistent set representation (u,v) == (v,u)
    return set(tuple(sorted(pair)) for pair in edges_array.T.tolist())

loaded_data = np.load("dataset/Arxiv.npz", allow_pickle=True)
    
train_edges_set = edges_to_set(loaded_data['train_edges'])
val_edges_set = edges_to_set(loaded_data['val_edges'])
test_edges_set = edges_to_set(loaded_data['test_edges'])
train_neg_set = edges_to_set(loaded_data['train_neg_edges'])
val_neg_set = edges_to_set(loaded_data['val_neg_edges'])
test_neg_set = edges_to_set(loaded_data['test_neg_edges'])
hop3_train_set = edges_to_set(loaded_data['hop3_train_edges'])
hop3_train_neg_set = edges_to_set(loaded_data['hop3_train_neg_edges'])
hop3_val_set = edges_to_set(loaded_data['hop3_val_edges'])
hop3_val_neg_set = edges_to_set(loaded_data['hop3_val_neg_edges'])

hop3_test_set = edges_to_set(loaded_data['hop3_test_edges'])
hop3_test_neg_set = edges_to_set(loaded_data['hop3_test_neg_edges'])
    
    # --- Core disjointness checks (Positives vs Negatives, within splits) ---
print("\n--- 1-Hop Split Disjointness ---")
print(f"Overlap train_edges vs val_edges: {len(train_edges_set.intersection(val_edges_set))}")
print(f"Overlap train_edges vs test_edges: {len(train_edges_set.intersection(test_edges_set))}")
print(f"Overlap val_edges vs test_edges: {len(val_edges_set.intersection(test_edges_set))}")

print("\n--- 3-Hop Split Disjointness ---")
print(f"Overlap hop3_train_edges vs hop3_test_edges: {len(hop3_train_set.intersection(hop3_test_set))}")
print(f"Overlap hop3_train_edges vs hop3_val_edges: {len(hop3_train_set.intersection(hop3_val_set))}")
print(f"Overlap hop3_val_edges vs hop3_test_edges: {len(hop3_val_set.intersection(hop3_test_set))}")
print(f"Overlap hop3_train_neg_edges vs hop3_test_neg_edges: {len(hop3_train_neg_set.intersection(hop3_test_neg_set))}")
print(f"Overlap hop3_train_neg_edges vs hop3_val_neg_edges: {len(hop3_train_neg_set.intersection(hop3_val_neg_set))}")
print(f"Overlap hop3_val_neg_edges vs hop3_test_neg_edges: {len(hop3_val_neg_set.intersection(hop3_test_neg_set))}")

print("\n--- Negative vs Positive Overlaps (CRITICAL) ---")
    # Overlap between 1-hop negatives and ANY positive (1-hop or 3-hop)
all_positive_edges_final_set = train_edges_set.union(val_edges_set).union(test_edges_set).union(hop3_train_set).union(hop3_val_set).union(hop3_test_set)
    
overlap_train_neg_pos = train_neg_set.intersection(all_positive_edges_final_set)
if overlap_train_neg_pos:
    print(f"⚠️ Overlap between 'train_neg_edges' and ANY positive edge: {len(overlap_train_neg_pos)}")
    print(list(overlap_train_neg_pos)[:5])
else:
    print("✅ No overlap between 'train_neg_edges' and ANY positive edge.")

overlap_val_neg_pos = val_neg_set.intersection(all_positive_edges_final_set)
if overlap_val_neg_pos:
    print(f"⚠️ Overlap between 'val_neg_edges' and ANY positive edge: {len(overlap_val_neg_pos)}")
    print(list(overlap_val_neg_pos)[:5])
else:
    print("✅ No overlap between 'val_neg_edges' and ANY positive edge.")

overlap_test_neg_pos = test_neg_set.intersection(all_positive_edges_final_set)
if overlap_test_neg_pos:
    print(f"⚠️ Overlap between 'test_neg_edges' and ANY positive edge: {len(overlap_test_neg_pos)}")
    print(list(overlap_test_neg_pos)[:5])
else:
    print("✅ No overlap between 'test_neg_edges' and ANY positive edge.")

overlap_hop3_train_neg_pos = hop3_train_neg_set.intersection(all_positive_edges_final_set)
if overlap_hop3_train_neg_pos:
    print(f"⚠️ Overlap between 'hop3_train_neg_edges' and ANY positive edge: {len(overlap_hop3_train_neg_pos)}")
    print(list(overlap_hop3_train_neg_pos)[:5])
else:
    print("✅ No overlap between 'hop3_train_neg_edges' and ANY positive edge.")

overlap_hop3_val_neg_pos = hop3_val_neg_set.intersection(all_positive_edges_final_set)
if overlap_hop3_train_neg_pos:
    print(f"⚠️ Overlap between 'hop3_val_neg_edges' and ANY positive edge: {len(overlap_hop3_val_neg_pos)}")
    print(list(overlap_hop3_val_neg_pos)[:5])
else:
    print("✅ No overlap between 'hop3_train_neg_edges' and ANY positive edge.")
        
overlap_hop3_test_neg_pos = hop3_test_neg_set.intersection(all_positive_edges_final_set)
if overlap_hop3_test_neg_pos:
    print(f"⚠️ Overlap between 'hop3_test_neg_edges' and ANY positive edge: {len(overlap_hop3_test_neg_pos)}")
    print(list(overlap_hop3_test_neg_pos)[:5])
else:
    print("✅ No overlap between 'hop3_test_neg_edges' and ANY positive edge.")

print("\n--- Finished all sanity checks ---")
