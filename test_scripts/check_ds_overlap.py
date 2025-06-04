import json
import os

def load_ids(json_file):
    """Load unique node pair IDs from a JSON file."""
    with open(json_file, 'r') as f:
        data = json.load(f)
    return {item['id'] for item in data}

def check_overlap(train_file, test_file, output_file):
    """Check for overlap between train and test IDs and save the report."""
    # Validate file existence
    if not os.path.exists(train_file):
        raise FileNotFoundError(f"Train file not found: {train_file}")
    if not os.path.exists(test_file):
        raise FileNotFoundError(f"Test file not found: {test_file}")

    # Load IDs
    train_ids = load_ids(train_file)
    test_ids = load_ids(test_file)

    # Find overlap
    overlap = train_ids & test_ids
    overlap_count = len(overlap)

    # Print results
    print(f"Checking overlap between train_all.json and test_all.json...")
    print(f"Number of overlapping node pairs: {overlap_count}")
    if overlap_count > 0:
        print(f"Overlapping IDs: {overlap}")

    # Save results to file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        f.write("Overlap Report between train_all.json and test_all.json\n\n")
        f.write(f"Number of overlapping node pairs: {overlap_count}\n")
        if overlap_count > 0:
            f.write(f"Overlapping IDs: {list(overlap)}\n")

    print(f"Overlap report saved to {output_file}")

def main():
    """Main function to check overlap."""
    # File paths
    train_file = "llm_pred/prompt_json/Arxiv/train_all.json"
    test_file = "llm_pred/prompt_json/Arxiv/new_test_all.json"
    output_file = "results/train_test_overlap_report.txt"

    # Check overlap
    check_overlap(train_file, test_file, output_file)

if __name__ == "__main__":
    main()