import numpy as np

def edges_to_set(edges_array):
    """Convert a (2, E) numpy array of edges to a set of sorted tuples (u, v)."""
    if edges_array.size == 0:
        return set()
    # Ensure tuples are sorted for consistent set representation (u,v) == (v,u)
    return set(tuple(sorted(pair)) for pair in edges_array.T.tolist())

def check_overlap(set1, name1, set2, name2):
    """Check and report overlap between two edge sets"""
    common = set1.intersection(set2)
    if len(common) > 0:
        print(f"⚠️ Found {len(common)} overlapping edges between '{name1}' and '{name2}':")
        print(list(common)[:5])
    else:
        print(f"✅ No overlap found between '{name1}' and '{name2}'")

ds_one = np.load("dataset/cora_v2.npz", allow_pickle=True)
    
ds_one_train_edges_set = edges_to_set(ds_one['train_edges'])
ds_one_val_edges_set = edges_to_set(ds_one['val_edges'])
ds_one_test_edges_set = edges_to_set(ds_one['test_edges'])
ds_one_train_neg_set = edges_to_set(ds_one['train_neg_edges'])
ds_one_val_neg_set = edges_to_set(ds_one['val_neg_edges'])
ds_one_test_neg_set = edges_to_set(ds_one['test_neg_edges'])
ds_one_hop3_train_set = edges_to_set(ds_one['hop3_train_edges'])
ds_one_hop3_train_neg_set = edges_to_set(ds_one['hop3_train_neg_edges'])

ds_one_hop3_test_set = edges_to_set(ds_one['hop3_test_edges'])
ds_one_hop3_test_neg_set = edges_to_set(ds_one['hop3_test_neg_edges'])

ds_two = np.load("dataset/cora_v3.npz", allow_pickle=True)
    
ds_two_train_edges_set = edges_to_set(ds_two['train_edges'])
ds_two_val_edges_set = edges_to_set(ds_two['val_edges'])
ds_two_test_edges_set = edges_to_set(ds_two['test_edges'])
ds_two_train_neg_set = edges_to_set(ds_two['train_neg_edges'])
ds_two_val_neg_set = edges_to_set(ds_two['val_neg_edges'])
ds_two_test_neg_set = edges_to_set(ds_two['test_neg_edges'])
ds_two_hop3_train_set = edges_to_set(ds_two['hop3_train_edges'])
ds_two_hop3_train_neg_set = edges_to_set(ds_two['hop3_train_neg_edges'])
ds_two_hop3_val_set = edges_to_set(ds_two['hop3_val_edges'])
ds_two_hop3_val_neg_set = edges_to_set(ds_two['hop3_val_neg_edges'])

ds_two_hop3_test_set = edges_to_set(ds_two['hop3_test_edges'])
ds_two_hop3_test_neg_set = edges_to_set(ds_two['hop3_test_neg_edges'])

check_overlap(ds_one_train_edges_set, 'ds_one_train_edges', ds_two_train_edges_set, 'ds_two_train_edges')
check_overlap(ds_one_train_edges_set, 'ds_one_train_edges', ds_two_val_edges_set, 'ds_two_val_edges')
check_overlap(ds_one_train_edges_set, 'ds_one_train_edges', ds_two_test_edges_set, 'ds_two_test_edges')
check_overlap(ds_one_train_neg_set, 'ds_one_train_neg_edges', ds_two_train_neg_set, 'ds_two_train_neg_edges')
check_overlap(ds_one_train_neg_set, 'ds_one_train_neg_edges', ds_two_val_neg_set, 'ds_two_val_neg_edges')
check_overlap(ds_one_train_neg_set, 'ds_one_train_neg_edges', ds_two_test_neg_set, 'ds_two_test_neg_edges')

check_overlap(ds_one_val_edges_set, 'ds_one_val_edges', ds_two_train_edges_set, 'ds_two_train_edges')
check_overlap(ds_one_val_edges_set, 'ds_one_val_edges', ds_two_val_edges_set, 'ds_two_val_edges')
check_overlap(ds_one_val_edges_set, 'ds_one_val_edges', ds_two_test_edges_set, 'ds_two_test_edges')
check_overlap(ds_one_val_neg_set, 'ds_one_val_neg_edges', ds_two_train_neg_set, 'ds_two_train_neg_edges')
check_overlap(ds_one_val_neg_set, 'ds_one_val_neg_edges', ds_two_val_neg_set, 'ds_two_val_neg_edges')
check_overlap(ds_one_val_neg_set, 'ds_one_val_neg_edges', ds_two_test_neg_set, 'ds_two_test_neg_edges')

check_overlap(ds_one_test_edges_set, 'ds_one_test_edges', ds_two_train_edges_set, 'ds_two_train_edges')
check_overlap(ds_one_test_edges_set, 'ds_one_test_edges', ds_two_val_edges_set, 'ds_two_val_edges')
check_overlap(ds_one_test_edges_set, 'ds_one_test_edges', ds_two_test_edges_set, 'ds_two_test_edges')
check_overlap(ds_one_test_neg_set, 'ds_one_test_neg_edges', ds_two_train_neg_set, 'ds_two_train_neg_edges')
check_overlap(ds_one_test_neg_set, 'ds_one_test_neg_edges', ds_two_val_neg_set, 'ds_two_val_neg_edges')
check_overlap(ds_one_test_neg_set, 'ds_one_test_neg_edges', ds_two_test_neg_set, 'ds_two_test_neg_edges')
    # Long-range train vs test
# check_overlap(hop3_train_set, 'hop3_train_edges', hop3_test_set, 'hop3_test_edges')
# check_overlap(hop3_train_neg_set, 'hop3_train_neg_edges', hop3_test_neg_set, 'hop3_test_neg_edges')

#     # Long-range vs 1-hop

# check_overlap(train_neg_set, 'train_neg_edges', hop3_train_set, 'hop3_train_edges')
# check_overlap(train_neg_set, 'train_neg_edges', hop3_test_set, 'hop3_test_edges')
# check_overlap(test_neg_set, 'test_neg_edges', hop3_train_set, 'hop3_train_edges')
# check_overlap(test_neg_set, 'test_neg_edges', hop3_test_set, 'hop3_test_edges')
# check_overlap(val_neg_set, 'val_neg_edges', hop3_train_set, 'hop3_train_edges')
# check_overlap(val_neg_set, 'test_neg_edges', hop3_test_set, 'hop3_test_edges')
# check_overlap(hop3_train_neg_set, 'hop3_train_neg_edges', test_set, 'test_edges')
# check_overlap(hop3_test_neg_set, 'hop3_test_neg_edges',test_set, 'test_edges' )
# check_overlap(hop3_train_neg_set, 'hop3_train_neg_edges', val_set, 'val_edges')
# check_overlap(hop3_test_neg_set, 'hop3_test_neg_edges',  val_set, 'val_edges')
    
# check_overlap(hop3_train_neg_set, 'hop3_train_neg_edges', train_set, 'train_edges')
# check_overlap(hop3_test_neg_set, 'hop3_test_neg_edges', train_set, 'train_edges')