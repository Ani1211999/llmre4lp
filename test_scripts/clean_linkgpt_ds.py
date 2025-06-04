import json
import re

def clean_text(text):
    """
    Remove <node_start>, <node>, <pairwise_start>, and <pairwise> tags from the text.
    """
    # Remove the specified tags using regex
    cleaned_text = re.sub(r'<node_start>|<node>|<pairwise_start>|<pairwise>', '', text)
    return cleaned_text

def clean_dataset(input_file, output_file):
    # Load the dataset
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Process each entry
    for entry in data:
        for conv in entry['conversations']:
            if conv['from'] == 'human':
                conv['value'] = clean_text(conv['value'])
    
    # Save the cleaned dataset
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"Cleaned dataset saved to {output_file}")

if __name__ == "__main__":
    input_file = "amazon_clothing_test_all.json"
    output_file = "amazon_clothing_test_all_cleaned.json"
    clean_dataset(input_file, output_file)