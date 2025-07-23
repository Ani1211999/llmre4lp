import json
import random

def sample_prompt(json_path, n=3):
    with open(json_path, "r") as f:
        prompts = json.load(f)
    print(f"Sampling {n} prompts from {json_path}")
    for p in random.sample(prompts, n):
        print("\n--- Prompt ---")
        for msg in p['conversations']:
            print(f"[{msg['from']}]: {msg['value']}")
        print("Ground Truth:", p['ground_truth'])

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python sample_prompts.py <path_to_prompt_json>")
        sys.exit(1)

    sample_prompt(sys.argv[1])