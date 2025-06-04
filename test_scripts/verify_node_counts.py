import json
import os
def verify_json_statistics(json_path, expected_positive, expected_negative, mode):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    total_prompts = len(data)
    positive_count = sum(1 for entry in data if entry['conversations'][1]['value'] == "Yes")
    negative_count = total_prompts - positive_count
    
    print(f"Statistics for {mode}:")
    print(f"Total prompts: {total_prompts} (expected: {expected_positive + expected_negative})")
    print(f"Positive prompts: {positive_count} (expected: {expected_positive})")
    print(f"Negative prompts: {negative_count} (expected: {expected_negative})")
    print(f"Ratio (neg:pos): {negative_count / positive_count:.1f} (expected: 5.0)")
    print("---")

# Verify each file
json_dir = "llm_pred/prompt_json/Arxiv/"
verify_json_statistics(os.path.join(json_dir, "train_all.json"), 3651, 18255, "train_all")
verify_json_statistics(os.path.join(json_dir, "val_all.json"), 456, 2280, "val_all")
verify_json_statistics(os.path.join(json_dir, "test_all.json"), 457, 2285, "new_test_all")
verify_json_statistics(os.path.join(json_dir, "long_range_infer.json"), 150, 750, "long_range_infer")