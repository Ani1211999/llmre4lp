import json

# Load the test_all.json dataset
def load_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

# Analyze class balance
def analyze_class_balance(dataset):
    labels = [entry['conversations'][1]['value'] for entry in dataset]
    yes_count = labels.count("Yes")
    no_count = labels.count("No")
    total = len(labels)
    yes_ratio = yes_count / total if total > 0 else 0
    no_ratio = no_count / total if total > 0 else 0
    
    # Print class balance
    print("\nClass Balance in test_all.json:")
    print(f"Total pairs: {total}")
    print(f"Number of 'Yes' labels: {yes_count} ({yes_ratio:.2%})")
    print(f"Number of 'No' labels: {no_count} ({no_ratio:.2%})")
    
    # Save to file
    class_balance = {
        "total_pairs": total,
        "yes_count": yes_count,
        "yes_ratio": yes_ratio,
        "no_count": no_count,
        "no_ratio": no_ratio
    }
    return class_balance

def main():
    # File path
    test_file = "/mnt/webscistorage/cc7738/ws_aniket/LLM_for_Het/LLM4HeG/llm_pred/prompt_json/Arxiv/new_test_all.json"
    
    # Load dataset
    test_data = load_json(test_file)
    
    # Analyze class balance
    class_balance= analyze_class_balance(test_data)
    print(class_balance)
if __name__ == "__main__":
    main()