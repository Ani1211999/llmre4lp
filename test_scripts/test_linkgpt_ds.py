import pickle

def load_pickle(file_name):
    """
    Load data from file_name with pickle format
    """
    with open(file_name, "rb") as f:
        data = pickle.load(f)
    return data

dataset_for_train = load_pickle('ft_yn_dataset.pkl')
total_pairs = sum(len(item) for item in dataset_for_train)

print(dataset_for_train[0])
print(len(dataset_for_train))
print(f"Total number of pairwise comparisons: {total_pairs}")
