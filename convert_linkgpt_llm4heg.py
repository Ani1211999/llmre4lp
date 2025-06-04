import pickle
import json
import os
import re
import random
from collections import defaultdict

def load_pickle(file_name):
    """
    Load data from file_name with pickle format
    """
    with open(file_name, "rb") as f:
        data = pickle.load(f)
    return data

def clean_text(text):
    """
    Remove <node_start>, <node>, <pairwise_start>, and <pairwise> tags from the text.
    """
    return re.sub(r'<node_start>|<node>|<pairwise_start>|<pairwise>', '', text)

def convert_to_json(input_file, output_file):
    # Load the dataset
    dataset = load_pickle(input_file)
    
    # Group pairs by source node
    pairs_by_node = defaultdict(list)
    for item in dataset:
        for pair in item:
            if not isinstance(pair, tuple) or len(pair) < 2:
                continue
            prompt, metadata = pair
            source_node = metadata.get('source_node', 'unknown')
            answer = prompt.split("Answer: ")[1].strip() if "Answer: " in prompt else "No"
            pairs_by_node[source_node].append((prompt, metadata, answer))
    
    # Prepare the output list
    output_data = []
    pair_idx = 0  # Global index for unique IDs
    
    # Process each source node
    for source_node, pairs in pairs_by_node.items():
        # Separate Yes and No pairs
        yes_pairs = [(prompt, metadata) for prompt, metadata, answer in pairs if answer == "Yes"]
        no_pairs = [(prompt, metadata) for prompt, metadata, answer in pairs if answer == "No"]
        
        # Keep all Yes pairs (should be exactly 1 per node)
        selected_pairs = yes_pairs.copy()
        
        # Select 2 No pairs (if available, otherwise all available)
        num_no_to_select = min(2, len(no_pairs))  # Select 2 No per Yes
        if num_no_to_select > 0 and len(no_pairs) >= num_no_to_select:
            selected_no_pairs = random.sample(no_pairs, num_no_to_select)
            selected_pairs.extend(selected_no_pairs)
        
        # Convert each selected pair to JSON format
        for prompt, metadata in selected_pairs:
            # Clean the prompt
            prompt = clean_text(prompt)
            
            # Extract source and target node IDs
            target_node = metadata.get('pairwise_target_id_ls', [f"target_{pair_idx}"])[0]
            item_id = f"{source_node}_{target_node}"
            
            # Split prompt into human and gpt parts
            human_part = prompt.split("Is this product also bought by the same user?")[0] + "Is this product also bought by the same user?"
            gpt_part = prompt.split("Answer: ")[1].strip() if "Answer: " in prompt else "No"
            
            # Create conversation structure
            conversation = [
                {
                    "from": "human",
                    "value": f"Background: Products in the Amazon clothing network. Each product is described by its title. Task: Predict whether two products are also bought by the same user based on their descriptions. {human_part}. Answer template: \"Yes\" or \"No\" for evaluation."
                },
                {
                    "from": "gpt",
                    "value": gpt_part
                }
            ]
            
            # Add to output data
            output_data.append({
                "id": item_id,
                "conversations": conversation
            })
            
            pair_idx += 1
    
    # Save to JSON file
    
    with open(output_file, "w") as fout:
        json.dump(output_data, fout, indent=2, ensure_ascii=False)
    
    print(f"Converted {len(output_data)} entries to {output_file}")
    
    # Print Yes/No distribution in the final dataset
    yes_count = sum(1 for entry in output_data if entry["conversations"][1]["value"] == "Yes")
    no_count = sum(1 for entry in output_data if entry["conversations"][1]["value"] == "No")
    print(f"Yes answers: {yes_count}")
    print(f"No answers: {no_count}")

if __name__ == "__main__":
    input_file = "eval_yn_dataset_0_examples.pkl"
    output_file = "amazon_clothing_test_all_reduced.json"
    convert_to_json(input_file, output_file)